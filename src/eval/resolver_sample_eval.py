"""Evaluate the player_join resolver against a random sample of real events.

Samples 20 historical NFL events spread across the 2023-2025 seasons (~7
Sundays worth of candidates, ~6-7 events sampled per season), pulls
player_anytime_td odds for each (regions=us, 10 credits/event), resolves
every outcome description against that game's two-team roster, and reports
match rate, every ambiguous case, every fuzzy match with its ratio, and any
resolved player whose position isn't QB/RB/WR/TE.

One-off evaluation script — not part of the ingest/feature pipeline.
"""

import random
from pathlib import Path

import pandas as pd
import requests

from src.ingest.odds_api import API_KEY, BASE_URL, REQUEST_TIMEOUT, SPORT_KEY
from src.features.player_join import build_roster_pool, resolve_player_name

REPO_ROOT = Path(__file__).resolve().parents[2]

# 3 Sundays per season, spread across the year, full slate each time.
SNAPSHOT_DATES = {
    2023: ["2023-09-17T14:00:00Z", "2023-11-05T14:00:00Z", "2023-12-24T14:00:00Z"],
    2024: ["2024-09-22T14:00:00Z", "2024-11-24T14:00:00Z", "2024-12-08T14:00:00Z"],
    2025: ["2025-09-21T14:00:00Z", "2025-11-02T14:00:00Z", "2025-12-21T14:00:00Z"],
}
EVENTS_PER_SEASON = {2023: 6, 2024: 7, 2025: 7}  # sums to 20
RANDOM_SEED = 42

NON_QRWT_LOG = []  # populated during resolution

credit_log = []


def _track_credits(label: str, resp: requests.Response) -> None:
    remaining = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    credit_log.append((label, resp.status_code, remaining, used))


def _get_events(snapshot_iso: str):
    resp = requests.get(
        f"{BASE_URL}/historical/sports/{SPORT_KEY}/events",
        params={"apiKey": API_KEY, "date": snapshot_iso},
        timeout=REQUEST_TIMEOUT,
    )
    _track_credits(f"events {snapshot_iso}", resp)
    if resp.status_code != 200:
        print(f"  ERROR fetching events for {snapshot_iso}: {resp.status_code} {resp.text}")
        return []
    payload = resp.json()
    events = payload.get("data") if isinstance(payload, dict) else payload
    return events or []


def _get_event_props(event_id: str, snapshot_iso: str):
    resp = requests.get(
        f"{BASE_URL}/historical/sports/{SPORT_KEY}/events/{event_id}/odds",
        params={
            "apiKey": API_KEY, "date": snapshot_iso, "regions": "us",
            "markets": "player_anytime_td", "oddsFormat": "decimal",
        },
        timeout=REQUEST_TIMEOUT,
    )
    _track_credits(f"props {event_id}", resp)
    if resp.status_code != 200:
        print(f"  ERROR fetching props for {event_id}: {resp.status_code} {resp.text}")
        return None
    return resp.json()


def _team_name_map() -> dict:
    import nfl_data_py as nfl
    td = nfl.import_team_desc()
    td = td[~td.team_abbr.isin(["LAR", "OAK", "SD", "STL"])]
    return dict(zip(td.team_name, td.team_abbr))


def sample_events() -> list:
    random.seed(RANDOM_SEED)
    sampled = []
    for season, dates in SNAPSHOT_DATES.items():
        candidates = []
        for snapshot_iso in dates:
            events = _get_events(snapshot_iso)
            for e in events:
                candidates.append({**e, "season": season, "snapshot_iso": snapshot_iso})
        n = min(EVENTS_PER_SEASON[season], len(candidates))
        chosen = random.sample(candidates, n)
        print(f"Season {season}: {len(candidates)} candidate events across {len(dates)} snapshots, sampled {n}")
        sampled.extend(chosen)
    return sampled


def evaluate() -> pd.DataFrame:
    team_map = _team_name_map()
    rosters = pd.read_parquet(REPO_ROOT / "data" / "raw" / "rosters.parquet")
    schedules = pd.read_parquet(REPO_ROOT / "data" / "raw" / "schedules.parquet")

    events = sample_events()
    print(f"\nTotal events sampled: {len(events)}\n")

    rows = []
    for i, event in enumerate(events, 1):
        home_full, away_full = event["home_team"], event["away_team"]
        home_abbr = team_map.get(home_full)
        away_abbr = team_map.get(away_full)
        print(f"[{i}/{len(events)}] {event['season']} {away_full} @ {home_full} "
              f"({event['commence_time']}) id={event['id']}")

        if home_abbr is None or away_abbr is None:
            print(f"  SKIP: unmapped team name(s) home={home_full!r} away={away_full!r}")
            continue

        payload = _get_event_props(event["id"], event["snapshot_iso"])
        if payload is None:
            continue

        data = payload.get("data") if isinstance(payload, dict) else payload
        bookmakers = (data or {}).get("bookmakers") or []
        # commence_time is UTC; nflverse's gameday is the US-local calendar date, which
        # differs for late Sunday/Monday night games that roll past midnight UTC.
        commence_time = pd.Timestamp((data or {}).get("commence_time", event["commence_time"])).tz_convert("America/New_York")
        game_type = "REG"

        # Roster weeks aren't dated directly; use nflverse schedules to find the week for this game.
        game_row = schedules[
            (schedules.season == event["season"])
            & (schedules.home_team == home_abbr)
            & (schedules.away_team == away_abbr)
            & (pd.to_datetime(schedules.gameday).dt.date == commence_time.date())
        ]
        if game_row.empty:
            print(f"  SKIP: couldn't find matching schedule row for {away_abbr}@{home_abbr} on {commence_time.date()}")
            continue
        week = int(game_row.iloc[0]["week"])

        pool = build_roster_pool(rosters, event["season"], week, home_abbr, away_abbr, game_type=game_type)
        if pool.empty:
            print(f"  SKIP: empty roster pool for {away_abbr}@{home_abbr} season={event['season']} week={week}")
            continue

        outcome_count = 0
        for bk in bookmakers:
            for m in bk.get("markets", []):
                for o in m.get("outcomes", []):
                    outcome_count += 1
                    result = resolve_player_name(o["description"], pool)
                    rows.append({
                        **result,
                        "event_id": event["id"], "season": event["season"], "week": week,
                        "home_team": home_abbr, "away_team": away_abbr,
                        "bookmaker": bk.get("key"),
                    })
        print(f"  {len(bookmakers)} bookmakers, {outcome_count} outcomes, roster pool={len(pool)}")

    return pd.DataFrame(rows)


def print_credit_log() -> None:
    print("\nCredit log")
    print("-" * 70)
    for label, status, remaining, used in credit_log:
        print(f"  {label:45} status={status} remaining={remaining} used={used}")


def print_report(results: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("Resolver evaluation report — 20-event sample, 2023-2025")
    print("=" * 60)

    is_player = results["status"] != "non_player"
    player_rows = results[is_player]
    resolved = player_rows["status"].isin(["exact", "fuzzy"])
    match_rate = resolved.mean() * 100 if len(player_rows) else 0.0

    print(f"\nTotal outcomes:        {len(results)}")
    print(f"  non_player entries:  {(results['status'] == 'non_player').sum()}")
    print(f"  player entries:      {len(player_rows)}")
    print(f"    resolved:          {resolved.sum()}")
    print(f"    ambiguous:         {(player_rows['status'] == 'ambiguous').sum()}")
    print(f"    unresolved:        {(player_rows['status'] == 'unresolved').sum()}")
    print(f"  MATCH RATE:          {match_rate:.1f}%")

    unique = results.drop_duplicates(subset=["event_id", "raw_name"])
    unique_players = unique[unique["status"] != "non_player"]
    unique_resolved = unique_players["status"].isin(["exact", "fuzzy"])
    unique_rate = unique_resolved.mean() * 100 if len(unique_players) else 0.0
    print(f"\nUnique (event, name) pairs: {len(unique)} ({len(unique_players)} players)")
    print(f"  unique match rate:        {unique_rate:.1f}%")

    print("\n--- Every ambiguous case ---")
    ambiguous = unique[unique["status"] == "ambiguous"]
    if ambiguous.empty:
        print("  none")
    else:
        for row in ambiguous.itertuples():
            print(f"  event={row.event_id} {row.away_team}@{row.home_team} raw_name={row.raw_name!r}")
            print(f"    {row.note}")

    print("\n--- Every fuzzy match (eyeball for wrongness) ---")
    fuzzy = unique[unique["status"] == "fuzzy"].copy()
    if fuzzy.empty:
        print("  none")
    else:
        fuzzy["ratio"] = fuzzy["method"].str.extract(r"ratio=([\d.]+)").astype(float)
        fuzzy = fuzzy.sort_values("ratio")
        for row in fuzzy.itertuples():
            print(f"  ratio={row.ratio:.2f}  {row.raw_name!r} -> {row.player_name!r} "
                  f"({row.team}/{row.position})  event={row.away_team}@{row.home_team}")

    print("\n--- Resolved to a non-skill-position player (not QB/RB/WR/TE) ---")
    resolved_rows = unique[unique["status"].isin(["exact", "fuzzy"])]
    off_position = resolved_rows[~resolved_rows["position"].isin(["QB", "RB", "WR", "TE"])]
    if off_position.empty:
        print("  none")
    else:
        for row in off_position.itertuples():
            print(f"  {row.raw_name!r} -> {row.player_name!r} position={row.position} team={row.team} "
                  f"status={row.status} event={row.away_team}@{row.home_team}")

    print("\n--- Every unresolved player-name case ---")
    unresolved = unique[unique["status"] == "unresolved"]
    if unresolved.empty:
        print("  none")
    else:
        for row in unresolved.itertuples():
            print(f"  event={row.event_id} {row.away_team}@{row.home_team} raw_name={row.raw_name!r} — {row.note}")

    print("=" * 60)


def main() -> None:
    results = evaluate()
    print_credit_log()
    print_report(results)


if __name__ == "__main__":
    main()
