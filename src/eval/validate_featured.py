"""Validate data/interim/featured.parquet: cross-check consensus closing
spreads against nflverse's spread_line, and characterize open-to-close
drift and observation coverage per event.

Pure offline analysis — no network calls, only cached parquet files.

Consensus home spread is book convention (home favourite = negative),
matching the rest of this project (see src/ingest/openers.py). nflverse's
spread_line uses the opposite convention (positive = home favoured), so a
correct match satisfies consensus_home_spread + spread_line == 0.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest.team_names import team_name_map

REPO_ROOT = Path(__file__).resolve().parents[2]

CLOSE_DIFF_ALERT_THRESHOLD = 3.0  # points
DRIFT_HIST_BINS = [-np.inf, -7, -5, -3, -1, 0, 1, 3, 5, 7, np.inf]


def load_data():
    featured = pd.read_parquet(REPO_ROOT / "data" / "interim" / "featured.parquet")
    schedules = pd.read_parquet(REPO_ROOT / "data" / "raw" / "schedules.parquet")
    return featured, schedules


def build_event_schedule_map(featured: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per (event_id, game_id): the odds-API's event_id joined to
    nflverse's game_id, home_team, away_team, season, spread_line.

    Matches via the two team names seen in the spreads market plus
    commence_time's US-local date (UTC->America/New_York, since late
    kickoffs roll to the next UTC day but schedules.gameday is US-local).

    Important: the odds API reissues a NEW event_id for the same real game
    when a far-future provisional listing is replaced closer to kickoff —
    confirmed on this data (e.g. JAX@TEN 2023 week 11 has one event_id with
    4 snapshots from Sep 7-10, abandoned, and a second with 12 snapshots
    from Nov 8-19 that actually tracks into the game). 265 of 1673 matched
    events are affected, concentrated in 2023. game_id (from schedules,
    always unique per real game) is the correct consolidation key —
    multiple event_ids legitimately map to one game_id here, and every
    downstream computation groups by game_id, not event_id.
    """
    spreads = featured[featured.market == "spreads"]
    team_map = team_name_map()

    per_event = spreads.groupby("event_id").agg(
        commence_time=("commence_time", "first"),
        teams=("team", lambda s: tuple(sorted(set(s)))),
    ).reset_index()

    n_before = len(per_event)
    per_event = per_event[per_event.teams.map(len) == 2].copy()
    n_dropped_team_count = n_before - len(per_event)

    per_event["commence_ts"] = pd.to_datetime(per_event.commence_time, utc=True)
    per_event["local_date"] = per_event.commence_ts.dt.tz_convert("America/New_York").dt.date

    abbrs = per_event.teams.map(lambda t: tuple(sorted(team_map.get(name) for name in t)))
    per_event["team_a_abbr"] = abbrs.map(lambda t: t[0])
    per_event["team_b_abbr"] = abbrs.map(lambda t: t[1])

    unmapped = per_event[per_event.team_a_abbr.isna() | per_event.team_b_abbr.isna()]
    per_event = per_event.dropna(subset=["team_a_abbr", "team_b_abbr"])

    sched = schedules.copy()
    sched["local_date"] = pd.to_datetime(sched.gameday).dt.date
    sched["team_a_abbr"] = sched[["home_team", "away_team"]].min(axis=1)
    sched["team_b_abbr"] = sched[["home_team", "away_team"]].max(axis=1)
    sched["commence_ts"] = pd.to_datetime(sched.gameday + " " + sched.gametime).dt.tz_localize(
        "America/New_York").dt.tz_convert("UTC")

    merged = per_event[["event_id", "local_date", "team_a_abbr", "team_b_abbr"]].merge(
        sched[["game_id", "season", "week", "home_team", "away_team", "spread_line",
               "commence_ts", "local_date", "team_a_abbr", "team_b_abbr"]],
        on=["local_date", "team_a_abbr", "team_b_abbr"], how="left",
    )

    n_unmatched = merged.spread_line.isna().sum()
    matched = merged.dropna(subset=["spread_line"]).copy()

    n_events_before_consolidation = matched.event_id.nunique()
    n_games_after_consolidation = matched.game_id.nunique()
    n_reissued = matched.groupby("game_id").event_id.nunique()
    n_reissued = (n_reissued > 1).sum()

    # home team's full name, needed to pick its `line` out of the spreads market
    inv_team_map = {}
    for full_name, abbr in team_map.items():
        inv_team_map.setdefault(abbr, full_name)
    matched["home_team_full"] = matched.home_team.map(inv_team_map)

    print(f"Event/schedule join: {len(per_event) + n_dropped_team_count} events in featured.parquet")
    if n_dropped_team_count:
        print(f"  dropped {n_dropped_team_count} events without exactly 2 distinct teams (data gap)")
    if len(unmapped):
        print(f"  dropped {len(unmapped)} events with an unmapped team name: "
              f"{sorted(set(t for pair in unmapped.teams for t in pair))}")
    if n_unmatched:
        print(f"  {n_unmatched} events had no matching schedules row (dropped)")
    print(f"  {n_events_before_consolidation} event_ids matched a schedule row")
    print(f"  -> consolidated to {n_games_after_consolidation} real games "
          f"({n_reissued} games had a reissued event_id, merged by game_id)\n")

    return matched[["event_id", "game_id", "commence_ts", "season", "week", "home_team", "away_team",
                     "home_team_full", "spread_line"]]


def compute_consensus_home_spreads(featured: pd.DataFrame, event_map: pd.DataFrame) -> pd.DataFrame:
    """game_id, snapshot_time, consensus_home_spread — mean of the home
    team's spread line across books, per snapshot. Snapshots from every
    event_id sharing a game_id (see build_event_schedule_map) are pooled
    into that game's single timeline."""
    spreads = featured[featured.market == "spreads"]
    game_lookup = event_map.set_index("event_id")["game_id"]
    home_lookup = event_map.set_index("event_id")["home_team_full"]

    tagged = spreads[spreads.event_id.isin(game_lookup.index)].copy()
    tagged["game_id"] = tagged.event_id.map(game_lookup)
    tagged["home_team_full"] = tagged.event_id.map(home_lookup)
    home_rows = tagged[tagged.team == tagged.home_team_full]

    consensus = home_rows.groupby(["game_id", "snapshot_time"]).line.mean().reset_index()
    consensus.columns = ["game_id", "snapshot_time", "consensus_home_spread"]
    consensus["snapshot_ts"] = pd.to_datetime(consensus.snapshot_time, utc=True)
    return consensus


def compute_close_spreads(consensus: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """game_id, season, close_spread, close_ts — consensus spread at the
    latest snapshot strictly before commence_time."""
    merged = consensus.merge(games[["game_id", "commence_ts", "season"]], on="game_id")
    before_kickoff = merged[merged.snapshot_ts < merged.commence_ts]
    latest = before_kickoff.sort_values("snapshot_ts").groupby("game_id").tail(1).copy()
    return latest.rename(columns={"consensus_home_spread": "close_spread", "snapshot_ts": "close_ts"})[
        ["game_id", "season", "close_spread", "close_ts"]]


def part1_close_cross_check(close: pd.DataFrame, games: pd.DataFrame) -> None:
    print("=" * 70)
    print("PART 1 — Close cross-check vs schedules.spread_line")
    print("=" * 70)

    latest = close.merge(games[["game_id", "spread_line"]], on="game_id").copy()
    n_no_pre_kickoff = games.game_id.nunique() - latest.game_id.nunique()

    latest["diff"] = (latest.close_spread + latest.spread_line).abs()

    print(f"Games with a pre-kickoff snapshot: {len(latest)} / {games.game_id.nunique()}")
    if n_no_pre_kickoff:
        print(f"  ({n_no_pre_kickoff} games had no snapshot before commence_time — excluded)")
    print()

    def _report(df, label):
        print(f"[{label}] n={len(df)}  median_abs_diff={df['diff'].median():.3f}  "
              f"count>{CLOSE_DIFF_ALERT_THRESHOLD}pt={int((df['diff'] > CLOSE_DIFF_ALERT_THRESHOLD).sum())} "
              f"({(df['diff'] > CLOSE_DIFF_ALERT_THRESHOLD).mean() * 100:.1f}%)")

    _report(latest, "ALL")
    print()
    for season, g in latest.groupby("season"):
        _report(g, f"season {season}")
    print()


def _print_histogram(values: pd.Series, bins) -> None:
    counts = pd.cut(values, bins=bins).value_counts().sort_index()
    max_count = counts.max() if len(counts) else 1
    for interval, count in counts.items():
        bar = "#" * int(50 * count / max_count) if max_count else ""
        print(f"  {str(interval):20} {count:5d}  {bar}")


def part2_drift_table(consensus: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    print("=" * 70)
    print("PART 2 — Open-to-close drift table")
    print("=" * 70)

    merged = consensus.merge(games[["game_id", "commence_ts", "season"]], on="game_id")
    before_kickoff = merged[merged.snapshot_ts <= merged.commence_ts]

    sorted_df = before_kickoff.sort_values("snapshot_ts")
    first = sorted_df.groupby("game_id").first()
    last = sorted_df.groupby("game_id").last()

    drift = pd.DataFrame({
        "game_id": first.index,
        "season": first["season"].values,
        "first_spread": first["consensus_home_spread"].values,
        "last_spread": last["consensus_home_spread"].values,
        "first_ts": first["snapshot_ts"].values,
        "last_ts": last["snapshot_ts"].values,
        "n_snapshots": sorted_df.groupby("game_id").size().values,
    })
    drift["drift"] = drift.last_spread - drift.first_spread
    drift["days_elapsed"] = (pd.to_datetime(drift.last_ts) - pd.to_datetime(drift.first_ts)).dt.total_seconds() / 86400

    single_snapshot = (drift.n_snapshots == 1).sum()
    print(f"Games analyzed: {len(drift)}  "
          f"({single_snapshot} have only 1 pre-kickoff snapshot — drift trivially 0 for these)\n")

    def _report(df, label):
        d = df["drift"]
        print(f"[{label}] n={len(d)}  mean={d.mean():+.3f}  median={d.median():+.3f}  "
              f"sd={d.std():.3f}  min={d.min():+.3f}  max={d.max():+.3f}  "
              f"frac_zero={(d == 0).mean() * 100:.1f}%")

    _report(drift, "ALL")
    print()
    for season, g in drift.groupby("season"):
        _report(g, f"season {season}")
    print()

    print("Drift histogram (all seasons, points; last-first, +=toward home team):")
    _print_histogram(drift["drift"], DRIFT_HIST_BINS)
    print()

    return drift


def part3_days_elapsed(drift: pd.DataFrame) -> None:
    print("=" * 70)
    print("PART 3 — Days elapsed between first and last observation")
    print("=" * 70)

    days = drift["days_elapsed"]

    print(f"n={len(days)}")
    print(days.describe().to_string())
    print()
    print("Distribution (days):")
    bins = [-0.001, 0, 1, 2, 3, 5, 7, 10, 14, 21, np.inf]
    _print_histogram(days, bins)
    print()


ELAPSED_BUCKETS = [(-0.001, 2, "0-2"), (2, 7, "3-7"), (7, 14, "8-14"),
                   (14, 30, "15-30"), (30, np.inf, "31+")]


def part4_drift_vs_elapsed(drift: pd.DataFrame) -> None:
    print("=" * 70)
    print("PART 4 — |drift| vs. days elapsed (open->close window)")
    print("=" * 70)

    abs_drift = drift["drift"].abs()
    days = drift["days_elapsed"]

    for lo, hi, label in ELAPSED_BUCKETS:
        mask = (days > lo) & (days <= hi)
        n = int(mask.sum())
        mean_abs = abs_drift[mask].mean() if n else float("nan")
        median_abs = abs_drift[mask].median() if n else float("nan")
        print(f"  [{label:6} days] n={n:4d}  mean|drift|={mean_abs:.3f}  median|drift|={median_abs:.3f}")
    print()


def compute_fixed_horizon_open(consensus: pd.DataFrame, games: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """game_id, season, open_spread, open_ts — consensus spread at the
    latest snapshot at or before (commence_time - horizon_days)."""
    merged = consensus.merge(games[["game_id", "commence_ts", "season"]], on="game_id")
    cutoff = merged.commence_ts - pd.Timedelta(days=horizon_days)
    eligible = merged[merged.snapshot_ts <= cutoff]
    latest = eligible.sort_values("snapshot_ts").groupby("game_id").tail(1).copy()
    return latest.rename(columns={"consensus_home_spread": "open_spread", "snapshot_ts": "open_ts"})[
        ["game_id", "season", "open_spread", "open_ts"]]


def part_fixed_horizon_drift(consensus: pd.DataFrame, close: pd.DataFrame, games: pd.DataFrame,
                              horizon_days: int) -> pd.DataFrame:
    print("=" * 70)
    print(f"PART — Fixed-horizon drift: close - (spread at kickoff-{horizon_days}d)")
    print("=" * 70)

    open_df = compute_fixed_horizon_open(consensus, games, horizon_days)
    n_total = games.game_id.nunique()
    n_no_horizon_snapshot = n_total - open_df.game_id.nunique()

    merged = open_df.merge(close[["game_id", "close_spread"]], on="game_id")
    n_no_close = open_df.game_id.nunique() - merged.game_id.nunique()
    merged["drift"] = merged.close_spread - merged.open_spread

    print(f"Games with a snapshot at/before kickoff-{horizon_days}d: {open_df.game_id.nunique()} / {n_total}")
    print(f"  ({n_no_horizon_snapshot} dropped — no snapshot that early)")
    if n_no_close:
        print(f"  ({n_no_close} further dropped — no pre-kickoff close either)")
    print(f"  {len(merged)} games analyzed\n")

    def _report(df, label):
        d = df["drift"]
        print(f"[{label}] n={len(d)}  mean={d.mean():+.3f}  median={d.median():+.3f}  "
              f"sd={d.std():.3f}  min={d.min():+.3f}  max={d.max():+.3f}  "
              f"frac_zero={(d == 0).mean() * 100:.1f}%")

    _report(merged, "ALL")
    print()
    for season, g in merged.groupby("season"):
        _report(g, f"season {season}")
    print()

    return merged


def main() -> None:
    featured, schedules = load_data()
    event_map = build_event_schedule_map(featured, schedules)
    consensus = compute_consensus_home_spreads(featured, event_map)

    # one row per real game — event_map has one row per (event_id, game_id)
    # pair, and multiple event_ids can share a game_id (reissued IDs)
    games = event_map.drop_duplicates(subset="game_id")

    close = compute_close_spreads(consensus, games)

    part1_close_cross_check(close, games)
    drift = part2_drift_table(consensus, games)
    part3_days_elapsed(drift)
    part4_drift_vs_elapsed(drift)
    part_fixed_horizon_drift(consensus, close, games, horizon_days=7)
    part_fixed_horizon_drift(consensus, close, games, horizon_days=3)


if __name__ == "__main__":
    main()
