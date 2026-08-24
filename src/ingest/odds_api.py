"""Validation-only probe of The Odds API's historical NFL endpoints.

Purpose: see the raw response shape for player_anytime_td props before
writing any parsing/caching logic. Each function below makes exactly ONE
request. Nothing here loops over events, dates, or seasons — running this
module end to end costs exactly 3 requests (1 events list + 2 odds calls).
"""

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

API_KEY = os.environ.get("ODDS_API_KEY")
if not API_KEY:
    raise RuntimeError("ODDS_API_KEY not found — check .env")

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_nfl"

# A Sunday morning in November 2024, before the early slate kicks off
# (13:00 ET / 18:00 UTC), so events are posted but games haven't started.
SNAPSHOT_ISO = "2024-11-10T14:00:00Z"

REQUEST_TIMEOUT = 30


def _print_credit_headers(resp: requests.Response) -> None:
    print(f"  x-requests-remaining: {resp.headers.get('x-requests-remaining')}")
    print(f"  x-requests-used:      {resp.headers.get('x-requests-used')}")


def list_historical_events(snapshot_iso: str):
    """GET /historical/sports/{sport}/events?date={snapshot_iso} — one request."""
    url = f"{BASE_URL}/historical/sports/{SPORT_KEY}/events"
    resp = requests.get(
        url,
        params={"apiKey": API_KEY, "date": snapshot_iso},
        timeout=REQUEST_TIMEOUT,
    )

    print(f"\n[list_historical_events] GET {resp.url}")
    print(f"  status: {resp.status_code}")
    _print_credit_headers(resp)

    if resp.status_code != 200:
        print(f"  ERROR body: {resp.text}")
        return None

    payload = resp.json()
    events = payload.get("data") if isinstance(payload, dict) else payload
    events = events or []

    print(f"  {len(events)} events:")
    for event in events:
        print(
            f"    id={event.get('id')} commence_time={event.get('commence_time')} "
            f"home={event.get('home_team')} away={event.get('away_team')}"
        )

    return payload


def fetch_event_props(event_id: str, snapshot_iso: str, regions: str):
    """GET /historical/sports/{sport}/events/{event_id}/odds — one request."""
    url = f"{BASE_URL}/historical/sports/{SPORT_KEY}/events/{event_id}/odds"
    resp = requests.get(
        url,
        params={
            "apiKey": API_KEY,
            "date": snapshot_iso,
            "regions": regions,
            "markets": "player_anytime_td",
            "oddsFormat": "decimal",
        },
        timeout=REQUEST_TIMEOUT,
    )

    print(f"\n[fetch_event_props] regions={regions} GET {resp.url}")
    print(f"  status: {resp.status_code}")
    _print_credit_headers(resp)

    if resp.status_code != 200:
        print(f"  ERROR body: {resp.text}")
        return None

    return resp.json()


def _describe_shape(value, max_depth: int = 3, depth: int = 0):
    if depth >= max_depth:
        return type(value).__name__
    if isinstance(value, dict):
        return {k: _describe_shape(v, max_depth, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        if not value:
            return "list[] (empty)"
        return [_describe_shape(value[0], max_depth, depth + 1)]
    return type(value).__name__


def summarize_region_response(region_label: str, payload) -> None:
    """Print bookmaker coverage, outcome shape, and top-level structure for one region."""
    print(f"\n=== Region: {region_label} ===")

    if payload is None:
        print("  No payload (request failed) — nothing to summarize.")
        return

    if not isinstance(payload, dict):
        print(f"  Unexpected top-level type: {type(payload).__name__}")
        print(f"  Raw: {payload!r}")
        return

    print("  Top-level keys:")
    for key, val in payload.items():
        if isinstance(val, (list, dict)):
            print(f"    {key}: {type(val).__name__} (len={len(val)})")
        else:
            print(f"    {key}: {type(val).__name__} = {val!r}")

    for ts_key in ("timestamp", "previous_timestamp", "next_timestamp", "commence_time"):
        if ts_key in payload:
            print(f"  snapshot field {ts_key!r} = {payload[ts_key]!r} (type={type(payload[ts_key]).__name__})")

    # The event-odds endpoint nests the event under "data"; bookmakers live
    # at data.bookmakers, not top-level. Handle both shapes defensively since
    # this is exactly the kind of thing that could differ across endpoints.
    event = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    bookmakers = event.get("bookmakers") if isinstance(event, dict) else None

    if not bookmakers:
        print("  No bookmakers with data for this region/snapshot/market — full shape:")
        print(json.dumps(_describe_shape(payload), indent=2, default=str))
        return

    print(f"\n  Bookmakers returned ({len(bookmakers)}):")
    example_outcome = None
    for bk in bookmakers:
        bk_key = bk.get("key", "?")
        markets = bk.get("markets") or []
        outcome_count = sum(len(m.get("outcomes") or []) for m in markets)
        print(f"    {bk_key}: {outcome_count} player outcomes across {len(markets)} market(s)")
        if example_outcome is None:
            for m in markets:
                outcomes = m.get("outcomes") or []
                if outcomes:
                    example_outcome = outcomes[0]
                    break

    print("\n  Shape of one outcome (keys and value types):")
    if example_outcome is None:
        print("    No outcomes found in any bookmaker/market.")
    else:
        for key, val in example_outcome.items():
            print(f"    {key}: {type(val).__name__} = {val!r}")


def main() -> None:
    events_payload = list_historical_events(SNAPSHOT_ISO)
    if not events_payload:
        print("\nNo events payload — stopping before making odds requests.")
        return

    events = events_payload.get("data") if isinstance(events_payload, dict) else events_payload
    events = events or []
    if not events:
        print("\nNo events returned for this snapshot — stopping.")
        return

    event_id = events[0]["id"]
    print(f"\nUsing first event for odds probe: {event_id}")

    uk_payload = fetch_event_props(event_id, SNAPSHOT_ISO, regions="uk")
    us_payload = fetch_event_props(event_id, SNAPSHOT_ISO, regions="us")

    summarize_region_response("uk", uk_payload)
    summarize_region_response("us", us_payload)


if __name__ == "__main__":
    main()
