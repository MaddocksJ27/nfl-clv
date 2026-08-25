"""Resolve free-text player names (e.g. from odds book prop descriptions) to
nflverse player_id, restricted to the two rosters playing in a given game.

Fetch/store lives in src.ingest; this module is pure matching logic over
already-cached roster data — no network calls.
"""

import difflib
import logging
import re
import unicodedata

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUFFIX_RE = re.compile(r"\s+\b(jr|sr|ii|iii|iv)\b\s*$")
PUNCTUATION_RE = re.compile(r"[.,']")
WHITESPACE_RE = re.compile(r"\s+")

# Prop books list team defense and "no touchdown" as pseudo-players in some
# markets. They will never match a roster and shouldn't be logged as if the
# matcher failed — flag them separately. Verified variants (from a 20-event
# sample across 2023-2025): "X D/ST", "X Defense", "X Defense/Special Teams",
# "No Touchdown", "No Scorer".
NON_PLAYER_RE = re.compile(
    r"\b(D/ST|Defense(/Special Teams)?)\b$|^No (Touchdown|Scorer)$", re.IGNORECASE,
)

# Some books disambiguate a common name with a parenthetical team tag, e.g.
# "Michael Thomas (NO)" or "Michael (Saints) Thomas" — strip parentheticals
# before normalizing so these hit the exact tier instead of relying on fuzzy.
PARENTHETICAL_RE = re.compile(r"\([^)]*\)")

FUZZY_CUTOFF = 0.82


def normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation/suffixes, collapse whitespace."""
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = PARENTHETICAL_RE.sub("", name)
    name = name.lower().replace("-", " ")
    name = PUNCTUATION_RE.sub("", name)
    name = SUFFIX_RE.sub("", name)
    name = WHITESPACE_RE.sub(" ", name).strip()
    return name


def build_roster_pool(
    rosters: pd.DataFrame, season: int, week: int, home_team: str, away_team: str,
    game_type: str = "REG",
) -> pd.DataFrame:
    """Restrict the roster to the two teams playing in this game, this week."""
    pool = rosters[
        (rosters.season == season)
        & (rosters.week == week)
        & (rosters.game_type == game_type)
        & (rosters.team.isin([home_team, away_team]))
    ][["player_id", "player_name", "position", "team"]].drop_duplicates(subset="player_id")

    pool = pool.assign(normalized_name=pool.player_name.map(normalize_name))
    return pool.reset_index(drop=True)


def resolve_player_name(raw_name: str, pool: pd.DataFrame, team_hint=None, position_hint=None) -> dict:
    """Match one raw name against a two-team roster pool.

    Returns a dict with at least {raw_name, status, player_id, player_name,
    team, position, method, note}. status is one of:
      exact       - unique normalized match
      fuzzy       - best candidate above FUZZY_CUTOFF, unambiguous
      ambiguous   - multiple equally-plausible candidates, hints didn't resolve it
      non_player  - looks like a team defense / "no touchdown" placeholder
      unresolved  - no candidate found
    """
    result = {
        "raw_name": raw_name, "status": None, "player_id": None, "player_name": None,
        "team": None, "position": None, "method": None, "note": None,
    }

    if NON_PLAYER_RE.search(raw_name or ""):
        result["status"] = "non_player"
        result["note"] = "team defense / no-touchdown placeholder, not a player"
        return result

    normalized = normalize_name(raw_name)
    if not normalized:
        result["status"] = "unresolved"
        result["note"] = "empty after normalization"
        logger.warning("UNRESOLVED %r -> empty after normalization", raw_name)
        return result

    def _apply_hints(candidates: pd.DataFrame) -> pd.DataFrame:
        narrowed = candidates
        if team_hint is not None:
            by_team = narrowed[narrowed.team == team_hint]
            if len(by_team) >= 1:
                narrowed = by_team
        if position_hint is not None:
            by_pos = narrowed[narrowed.position == position_hint]
            if len(by_pos) >= 1:
                narrowed = by_pos
        return narrowed

    exact = pool[pool.normalized_name == normalized]
    if len(exact) == 1:
        row = exact.iloc[0]
        result.update(status="exact", player_id=row.player_id, player_name=row.player_name,
                       team=row.team, position=row.position, method="normalized exact match")
        return result

    if len(exact) > 1:
        narrowed = _apply_hints(exact)
        if len(narrowed) == 1:
            row = narrowed.iloc[0]
            result.update(status="exact", player_id=row.player_id, player_name=row.player_name,
                           team=row.team, position=row.position,
                           method="normalized exact match, disambiguated by team/position hint")
            return result
        candidates = ", ".join(f"{r.player_name} ({r.team}/{r.position})" for r in exact.itertuples())
        result["status"] = "ambiguous"
        result["note"] = f"{len(exact)} exact candidates, hints did not narrow to one: {candidates}"
        logger.warning("AMBIGUOUS %r -> %s", raw_name, result["note"])
        return result

    # No exact match — fall back to fuzzy matching within the pool only.
    choices = pool.normalized_name.tolist()
    close = difflib.get_close_matches(normalized, choices, n=3, cutoff=FUZZY_CUTOFF)
    if not close:
        result["status"] = "unresolved"
        result["note"] = "no exact or fuzzy match within game roster pool"
        logger.warning("UNRESOLVED %r -> no candidate above cutoff=%.2f in roster pool", raw_name, FUZZY_CUTOFF)
        return result

    best = close[0]
    best_candidates = pool[pool.normalized_name == best]
    if len(best_candidates) > 1:
        narrowed = _apply_hints(best_candidates)
        if len(narrowed) != 1:
            candidates = ", ".join(f"{r.player_name} ({r.team}/{r.position})" for r in best_candidates.itertuples())
            result["status"] = "ambiguous"
            result["note"] = f"fuzzy match {best!r} has {len(best_candidates)} candidates: {candidates}"
            logger.warning("AMBIGUOUS %r -> %s", raw_name, result["note"])
            return result
        best_candidates = narrowed

    row = best_candidates.iloc[0]
    ratio = difflib.SequenceMatcher(None, normalized, best).ratio()
    result.update(status="fuzzy", player_id=row.player_id, player_name=row.player_name,
                   team=row.team, position=row.position, method=f"fuzzy match (ratio={ratio:.2f})")
    logger.info("FUZZY %r -> %r (ratio=%.2f)", raw_name, row.player_name, ratio)
    return result


def resolve_player_names(raw_names, pool: pd.DataFrame, team_hint=None, position_hint=None) -> pd.DataFrame:
    results = [resolve_player_name(name, pool, team_hint, position_hint) for name in raw_names]
    return pd.DataFrame(results)


def _print_report(results: pd.DataFrame) -> None:
    total = len(results)
    is_player = results["status"] != "non_player"
    player_rows = results[is_player]
    resolved = player_rows["status"].isin(["exact", "fuzzy"])

    print("\nPlayer name resolution report")
    print("=" * 50)
    print(f"Total names tested:        {total}")
    print(f"  non_player (D/ST etc.):  {(results['status'] == 'non_player').sum()}")
    print(f"  player-name entries:     {len(player_rows)}")
    print(f"    resolved (exact+fuzzy): {resolved.sum()}")
    print(f"    ambiguous:              {(player_rows['status'] == 'ambiguous').sum()}")
    print(f"    unresolved:             {(player_rows['status'] == 'unresolved').sum()}")
    match_rate = resolved.mean() * 100 if len(player_rows) else 0.0
    print(f"  match rate (of player-name entries): {match_rate:.1f}%")

    unique = results.drop_duplicates(subset="raw_name")
    unique_players = unique[unique["status"] != "non_player"]
    unique_resolved = unique_players["status"].isin(["exact", "fuzzy"])
    unique_rate = unique_resolved.mean() * 100 if len(unique_players) else 0.0
    print(f"\nUnique names tested: {len(unique)} ({len(unique_players)} players)")
    print(f"  unique match rate: {unique_rate:.1f}%")

    failures = unique[unique["status"].isin(["ambiguous", "unresolved"])]
    print(f"\nFailures ({len(failures)}):")
    if failures.empty:
        print("  none")
    else:
        for row in failures.itertuples():
            print(f"  [{row.status}] {row.raw_name!r} — {row.note}")
    print("=" * 50)


def _demo_normalization_robustness(pool: pd.DataFrame) -> None:
    """The real validation pull happens to match the roster verbatim, so it
    never exercises suffix/accent/fuzzy handling. Prove those paths work
    against synthetic variants of real roster names, kept separate from the
    real match-rate numbers above.
    """
    print("\nSynthetic robustness check (not part of the match-rate above)")
    print("-" * 50)
    cases = [
        "TYRONE TRACY",           # case + dropped suffix
        "Tyrone Tracy, Jr.",      # comma-suffix variant
        "Théo Johnson",           # accent
        "Ja Tavion Sanders",      # dropped apostrophe
        "Bryce Ford Wheaton",     # hyphen -> space
        "Bryce Fnord-Wheaton",    # misspelling -> should hit fuzzy tier
    ]
    for raw in cases:
        result = resolve_player_name(raw, pool)
        print(f"  {raw!r:32} -> status={result['status']:10} matched={result['player_name']!r} method={result['method']}")


def main() -> None:
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    fixture_path = repo_root / "data" / "interim" / "validation_player_names.json"
    fixture = json.loads(fixture_path.read_text())

    rosters = pd.read_parquet(repo_root / "data" / "raw" / "rosters.parquet")
    pool = build_roster_pool(
        rosters, fixture["season"], fixture["week"], fixture["home_team"], fixture["away_team"],
    )
    print(f"Roster pool: {len(pool)} players ({fixture['home_team']} vs {fixture['away_team']}, "
          f"season {fixture['season']} week {fixture['week']})")

    results = resolve_player_names(fixture["raw_names"], pool)
    _print_report(results)
    _demo_normalization_robustness(pool)


if __name__ == "__main__":
    main()
