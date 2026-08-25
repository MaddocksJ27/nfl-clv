"""Bulk-pull historical NFL spreads/totals via the odds-API bulk historical
endpoint (all events at a snapshot in one call — not per-event).

Date range: 2020-06-01 through 2025-02-28 (5 seasons, 2020-2024).

Snapshot cadence: daily at 12:00 UTC across each season's game window, all
five seasons (2020-2024) — no hourly pre-kickoff cadence.
Processed newest-first (reverse chronological) — 2024 season first, 2020
season last.

Cost is 10 credits x markets x regions PER SNAPSHOT regardless of game
count (2 markets x 1 region = 20 credits/snapshot here) — cheap per
snapshot, but the snapshot count is large; see the dry run cost estimate
before running in full.

Every snapshot's raw JSON is cached to data/raw/featured/{snapshot}.json
before parsing — a cached snapshot is never refetched, so this is safe to
interrupt and resume.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import requests

from src.ingest.odds_api import API_KEY, BASE_URL, REQUEST_TIMEOUT, SPORT_KEY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "featured"
OUTPUT_PATH = REPO_ROOT / "data" / "interim" / "featured.parquet"

SEASONS = [2020, 2021, 2022, 2023, 2024]
RANGE_START = pd.Timestamp("2020-06-01", tz="UTC")
RANGE_END = pd.Timestamp("2025-02-28", tz="UTC")

MARKETS = "spreads,totals"
REGION = "us"
ODDS_FORMAT = "decimal"

LOG_EVERY_N = 25
CREDIT_FLOOR = 20_000
CREDITS_PER_SNAPSHOT = 20  # 10 x 2 markets x 1 region, per the task's cost model


def build_snapshot_schedule() -> pd.DataFrame:
    """One row per unique snapshot timestamp: daily 12:00 UTC across each
    season's game window, all seasons, no hourly pre-kickoff cadence.

    Returns a DataFrame sorted newest-first (reverse chronological — the
    processing order requested: most recent season first, oldest last)
    with an ISO string column for the API's `date` param.
    """
    schedules = pd.read_parquet(REPO_ROOT / "data" / "raw" / "schedules.parquet")
    games = schedules[schedules.season.isin(SEASONS)].copy()

    timestamps = set()

    for season in SEASONS:
        s = games[games.season == season]
        start = pd.Timestamp(s.gameday.min(), tz="UTC")
        end = pd.Timestamp(s.gameday.max(), tz="UTC")
        n_days = (end - start).days + 1
        for d in range(n_days):
            timestamps.add(start + pd.Timedelta(days=d, hours=12))

    ts_series = pd.Series(sorted(timestamps, reverse=True), name="snapshot_ts")
    ts_series = ts_series[(ts_series >= RANGE_START) & (ts_series <= RANGE_END)]

    out = ts_series.to_frame()
    out["snapshot_iso"] = out["snapshot_ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out.reset_index(drop=True)


def _get(url: str, params: dict):
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    remaining = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    return resp, remaining, used


def _cache_path(snapshot_iso: str) -> Path:
    safe = snapshot_iso.replace(":", "")
    return RAW_DIR / f"{safe}.json"


def fetch_and_cache_snapshot(snapshot_iso: str):
    """Returns (payload, status, remaining) where status is 'cached', 'fetched', or 'error'."""
    path = _cache_path(snapshot_iso)
    if path.exists():
        return json.loads(path.read_text()), "cached", None

    resp, remaining, used = _get(
        f"{BASE_URL}/historical/sports/{SPORT_KEY}/odds",
        {
            "apiKey": API_KEY, "date": snapshot_iso, "regions": REGION,
            "markets": MARKETS, "oddsFormat": ODDS_FORMAT,
        },
    )
    if resp.status_code != 200:
        logger.error("Fetch failed for snapshot %s: %s %s", snapshot_iso, resp.status_code, resp.text)
        return None, "error", remaining

    payload = resp.json()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload, "fetched", remaining


def parse_snapshot_payload(payload: dict, snapshot_iso: str) -> list:
    """Long-format rows: snapshot_time, event_id, commence_time, book,
    market, team, line, price. `team` holds the outcome name verbatim —
    a team name for the spreads market, "Over"/"Under" for totals."""
    events = payload.get("data") if isinstance(payload, dict) else payload
    events = events or []

    rows = []
    for event in events:
        event_id = event.get("id")
        commence_time = event.get("commence_time")
        for bk in event.get("bookmakers", []):
            book = bk.get("key")
            for m in bk.get("markets", []):
                market = m.get("key")
                for o in m.get("outcomes", []):
                    rows.append({
                        "snapshot_time": snapshot_iso,
                        "event_id": event_id,
                        "commence_time": commence_time,
                        "book": book,
                        "market": market,
                        "team": o.get("name"),
                        "line": o.get("point"),
                        "price": o.get("price"),
                    })
    return rows


def run(schedule: pd.DataFrame, limit: int = None) -> pd.DataFrame:
    to_process = schedule.head(limit) if limit is not None else schedule

    all_rows = []
    last_remaining = None
    n_fetched = n_cached = n_errors = 0
    i = 0

    for i, row in enumerate(to_process.itertuples(), 1):
        if last_remaining is not None and last_remaining < CREDIT_FLOOR:
            logger.warning("Credit floor reached (remaining=%d < %d) — stopping before snapshot %s",
                            last_remaining, CREDIT_FLOOR, row.snapshot_iso)
            break

        payload, status, remaining = fetch_and_cache_snapshot(row.snapshot_iso)
        if remaining is not None:
            last_remaining = int(remaining)

        if status == "cached":
            n_cached += 1
        elif status == "fetched":
            n_fetched += 1
        else:
            n_errors += 1
            continue

        all_rows.extend(parse_snapshot_payload(payload, row.snapshot_iso))

        if i % LOG_EVERY_N == 0:
            logger.info("[%d/%d] fetched=%d cached=%d errors=%d remaining=%s",
                        i, len(to_process), n_fetched, n_cached, n_errors, last_remaining)

    print(f"\nDone: {i}/{len(to_process)} snapshots processed "
          f"(fetched={n_fetched}, cached={n_cached}, errors={n_errors}), "
          f"last known remaining={last_remaining}")

    return pd.DataFrame(all_rows)


def main(limit: int = None) -> pd.DataFrame:
    schedule = build_snapshot_schedule()
    projected_credits = len(schedule) * CREDITS_PER_SNAPSHOT
    print(f"Snapshot schedule: {len(schedule)} unique snapshots, "
          f"projected cost ~{projected_credits:,} credits at {CREDITS_PER_SNAPSHOT}/snapshot")

    results = run(schedule, limit=limit)

    if not results.empty:
        if limit is not None:
            print(f"\n(dry run with limit={limit} — not writing to {OUTPUT_PATH})")
        else:
            (REPO_ROOT / "data" / "interim").mkdir(parents=True, exist_ok=True)
            results.to_parquet(OUTPUT_PATH, index=False)
            print(f"Wrote {len(results)} rows to {OUTPUT_PATH}")

    return results


if __name__ == "__main__":
    main()
