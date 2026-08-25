"""Bulk-pull closing player-prop odds for every NFL game, 2023 week 1
through 2025 week 22, and parse into a long-format parquet.

Snapshot per game: commence_time - 10 minutes (closing price, the
calibration target). Every event's raw JSON is cached to
data/raw/props/{event_id}.json before parsing — an event already cached is
never refetched, so this script is safe to interrupt and resume.

Credit-conscious by design: logs x-requests-remaining every LOG_EVERY_N
events and hard-stops (no further requests) once remaining credits drop
below CREDIT_FLOOR, printing progress so far.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import requests

from src.features.player_join import build_roster_pool, resolve_player_name
from src.ingest.odds_api import API_KEY, BASE_URL, REQUEST_TIMEOUT, SPORT_KEY
from src.ingest.team_names import team_name_map

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PROPS_DIR = REPO_ROOT / "data" / "raw" / "props"
EVENT_INDEX_PATH = RAW_PROPS_DIR / "_event_index.parquet"
OUTPUT_PATH = REPO_ROOT / "data" / "interim" / "props.parquet"

SEASONS = [2023, 2024, 2025]
WEEKS = range(1, 23)  # 1-18 regular season, 19-22 postseason (WC/DIV/CONF/SB)

MARKETS = "player_anytime_td,player_receptions,player_rush_attempts,player_rush_yds,player_reception_yds,player_pass_yds"
REGION = "us"
ODDS_FORMAT = "decimal"

LOG_EVERY_N = 25
CREDIT_FLOOR = 25_000

# nflverse's rosters.parquet uses these game_type codes for postseason weeks
# rather than "REG" — weeks 1-18 are regular season.
POSTSEASON_GAME_TYPE = {19: "WC", 20: "DIV", 21: "CON", 22: "SB"}


def game_type_for_week(week: int) -> str:
    return POSTSEASON_GAME_TYPE.get(week, "REG")


class CreditFloorReached(Exception):
    pass


def build_game_list() -> pd.DataFrame:
    """One row per game, 2023 wk1 - 2025 wk22, with UTC commence_time and
    the closing snapshot (commence_time - 10min) as an ISO8601 string."""
    schedules = pd.read_parquet(REPO_ROOT / "data" / "raw" / "schedules.parquet")
    games = schedules[schedules.season.isin(SEASONS) & schedules.week.isin(WEEKS)].copy()

    local_dt = pd.to_datetime(games["gameday"] + " " + games["gametime"])
    games["commence_time"] = local_dt.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    games["snapshot_ts"] = games["commence_time"] - pd.Timedelta(minutes=10)
    games["snapshot_iso"] = games["snapshot_ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    games = games.sort_values(["season", "week", "commence_time"]).reset_index(drop=True)
    return games[["game_id", "season", "week", "home_team", "away_team", "commence_time", "snapshot_iso"]]


def _get(url: str, params: dict):
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    remaining = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    return resp, remaining, used


def discover_event_ids(games: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    """Resolve each (season, week, home_team, away_team) row to an odds-API
    event_id, one events-list call per (season, week). Cached to disk so
    resuming never re-does this discovery pass."""
    if EVENT_INDEX_PATH.exists() and not force_refresh:
        logger.info("Loading cached event index from %s", EVENT_INDEX_PATH)
        return pd.read_parquet(EVENT_INDEX_PATH)

    team_map = team_name_map()
    rows = []

    for (season, week), group in games.groupby(["season", "week"]):
        anchor = (group["commence_time"].min() - pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        resp, remaining, used = _get(
            f"{BASE_URL}/historical/sports/{SPORT_KEY}/events",
            {"apiKey": API_KEY, "date": anchor},
        )
        if resp.status_code != 200:
            logger.error("Events list failed for season=%s week=%s: %s %s", season, week, resp.status_code, resp.text)
            continue

        payload = resp.json()
        events = payload.get("data") if isinstance(payload, dict) else payload
        events = events or []

        by_key = {}
        for e in events:
            home_abbr = team_map.get(e.get("home_team"))
            away_abbr = team_map.get(e.get("away_team"))
            if home_abbr is None or away_abbr is None:
                continue
            local_date = pd.Timestamp(e["commence_time"]).tz_convert("America/New_York").date()
            by_key[(home_abbr, away_abbr, local_date)] = e["id"]

        for row in group.itertuples():
            local_date = row.commence_time.tz_convert("America/New_York").date()
            key = (row.home_team, row.away_team, local_date)
            event_id = by_key.get(key)
            if event_id is None:
                logger.warning("No event_id match for %s season=%s week=%s (%s@%s)",
                                row.game_id, season, week, row.away_team, row.home_team)
            rows.append({"game_id": row.game_id, "season": season, "week": week,
                         "home_team": row.home_team, "away_team": row.away_team,
                         "commence_time": row.commence_time, "snapshot_iso": row.snapshot_iso,
                         "event_id": event_id})

        logger.info("season=%s week=%s: matched %d/%d games (remaining=%s)",
                    season, week, sum(1 for r in rows[-len(group):] if r["event_id"]), len(group), remaining)

    index = pd.DataFrame(rows)
    RAW_PROPS_DIR.mkdir(parents=True, exist_ok=True)
    index.to_parquet(EVENT_INDEX_PATH, index=False)
    return index


def _cache_path(event_id: str) -> Path:
    return RAW_PROPS_DIR / f"{event_id}.json"


def fetch_and_cache_event(event_id: str, snapshot_iso: str):
    """Returns (payload, status, remaining) where status is 'cached', 'fetched', or 'error'."""
    path = _cache_path(event_id)
    if path.exists():
        return json.loads(path.read_text()), "cached", None

    resp, remaining, used = _get(
        f"{BASE_URL}/historical/sports/{SPORT_KEY}/events/{event_id}/odds",
        {
            "apiKey": API_KEY, "date": snapshot_iso, "regions": REGION,
            "markets": MARKETS, "oddsFormat": ODDS_FORMAT,
        },
    )
    if resp.status_code != 200:
        logger.error("Props fetch failed for %s: %s %s", event_id, resp.status_code, resp.text)
        return None, "error", remaining

    payload = resp.json()
    RAW_PROPS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload, "fetched", remaining


def parse_event_payload(payload: dict, event_id: str, commence_time, pool: pd.DataFrame) -> list:
    """Long-format rows: event_id, commence_time, book, market, player_id,
    player_name, position, line, side, price."""
    data = payload.get("data") if isinstance(payload, dict) else payload
    bookmakers = (data or {}).get("bookmakers") or []

    rows = []
    for bk in bookmakers:
        book = bk.get("key")
        for m in bk.get("markets", []):
            market = m.get("key")
            for o in m.get("outcomes", []):
                raw_name = o.get("description")
                result = resolve_player_name(raw_name, pool) if raw_name else {
                    "player_id": None, "player_name": None, "position": None,
                }
                rows.append({
                    "event_id": event_id,
                    "commence_time": commence_time,
                    "book": book,
                    "market": market,
                    "player_id": result["player_id"],
                    "player_name": result["player_name"] or raw_name,
                    "position": result["position"],
                    "line": o.get("point"),
                    "side": o.get("name"),
                    "price": o.get("price"),
                })
    return rows


def run(index: pd.DataFrame, rosters: pd.DataFrame, limit: int = None) -> pd.DataFrame:
    to_process = index[index.event_id.notna()]
    if limit is not None:
        to_process = to_process.head(limit)

    all_rows = []
    last_remaining = None
    n_fetched = n_cached = n_errors = 0

    for i, row in enumerate(to_process.itertuples(), 1):
        if last_remaining is not None and last_remaining < CREDIT_FLOOR:
            logger.warning("Credit floor reached (remaining=%d < %d) — stopping before event %s",
                            last_remaining, CREDIT_FLOOR, row.event_id)
            break

        payload, status, remaining = fetch_and_cache_event(row.event_id, row.snapshot_iso)
        if remaining is not None:
            last_remaining = int(remaining)

        if status == "cached":
            n_cached += 1
        elif status == "fetched":
            n_fetched += 1
        else:
            n_errors += 1
            continue

        pool = build_roster_pool(rosters, row.season, row.week, row.home_team, row.away_team,
                                  game_type=game_type_for_week(row.week))
        all_rows.extend(parse_event_payload(payload, row.event_id, row.commence_time, pool))

        if i % LOG_EVERY_N == 0:
            logger.info("[%d/%d] fetched=%d cached=%d errors=%d remaining=%s",
                        i, len(to_process), n_fetched, n_cached, n_errors, last_remaining)

    print(f"\nDone: {i}/{len(to_process)} events processed "
          f"(fetched={n_fetched}, cached={n_cached}, errors={n_errors}), "
          f"last known remaining={last_remaining}")

    return pd.DataFrame(all_rows)


def main(limit: int = None) -> pd.DataFrame:
    games = build_game_list()
    print(f"Game list: {len(games)} games, {games.season.nunique()} seasons")

    index = discover_event_ids(games)
    matched = index.event_id.notna().sum()
    print(f"Event index: {matched}/{len(index)} games matched to an event_id")

    rosters = pd.read_parquet(REPO_ROOT / "data" / "raw" / "rosters.parquet")
    results = run(index, rosters, limit=limit)

    if not results.empty:
        (REPO_ROOT / "data" / "interim").mkdir(parents=True, exist_ok=True)
        if limit is not None:
            print(f"\n(dry run with limit={limit} — not writing to {OUTPUT_PATH})")
        else:
            results.to_parquet(OUTPUT_PATH, index=False)
            print(f"Wrote {len(results)} rows to {OUTPUT_PATH}")

    return results


if __name__ == "__main__":
    main()
