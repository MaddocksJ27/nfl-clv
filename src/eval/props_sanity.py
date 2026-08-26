"""Sanity-check the anytime-TD props market: margin-based de-vig and
calibration against actual touchdown scorers.

Pure offline analysis — no network calls, only cached parquet files.

## De-vig method (v2 — replaces the earlier Shin/proportional pass)

Per-outcome (Yes/No) de-vig doesn't work here: only 0.9% of anytime-TD rows
have a "No" side priced at all. The first version instead summed all
players' Yes-prices in a (game_id, book) and de-vigged them as if mutually
exclusive (Shin or proportional normalization to 1.0) — mechanically valid,
but the wrong model: multiple players DO score in the same game, so forcing
the field to sum to 1 discards real information and (as found) produced
probabilities calibrated ~4x too low across every decile and position.

This version instead estimates a per-leg MULTIPLICATIVE margin:
    m(game, book) = sum_i(raw implied prob) / expected_distinct_scorers(game)
    devigged_prob_i = raw_prob_i / m(game, book)
`expected_distinct_scorers` comes from a simple OOS linear fit (walk-forward
by season, train strictly-earlier seasons) of actual distinct scorers on
the game's own total line alone — a genuine held-out prediction, not the
realized count itself, so m isn't circularly defined from the answer.

By construction, sum_i(devigged_prob_i) == expected_distinct_scorers(game)
for every book on a given game (m absorbs whatever that book's own raw sum
was) — every book's de-vigged field lands on the same model-implied count
for that game. That's intentional, not a bug: it's what makes different
books' de-vigged probabilities comparable at all, and it's a genuinely
game-varying number now (unlike v1's flat 1.0), driven by the total line.

## Scratch definition (v2 — replaces the pbp-participation check)

v1 flagged a prop row as a "scratch" if the player never appeared in any of
several pbp participation columns for that game — 22.8% of rows. Checking
who those actually were showed the check was overbroad: the top names
(Reggie Gilliam, Chris Manhertz, Patrick Ricard, C.J. Ham) are blocking
FBs/TEs who dressed and played but simply recorded zero offensive touches
that game — not scratches. This version uses snap_counts.parquet directly:
a player is a scratch only if he has no snap-count row for that game, or
offense_snaps == 0. snap_counts only carries a PFR player id (not the GSIS
id used everywhere else), so matching is by normalized name + team + season
+ week, reusing src.features.player_join's normalizer.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.features.player_join import normalize_name

REPO_ROOT = Path(__file__).resolve().parents[2]
PROPS_PATH = REPO_ROOT / "data" / "interim" / "props.parquet"
EVENT_INDEX_PATH = REPO_ROOT / "data" / "raw" / "props" / "_event_index.parquet"
PBP_PATH = REPO_ROOT / "data" / "raw" / "pbp.parquet"
ROSTERS_PATH = REPO_ROOT / "data" / "raw" / "rosters.parquet"
SCHEDULES_PATH = REPO_ROOT / "data" / "raw" / "schedules.parquet"
SNAP_COUNTS_PATH = REPO_ROOT / "data" / "raw" / "snap_counts.parquet"

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
ACTUAL_SCORER_SEASONS = (2023, 2024, 2025)


def load_td_props() -> pd.DataFrame:
    """anytime-TD, Yes side, QB/RB/WR/TE only, with game_id/season/week
    joined in from the event index."""
    props = pd.read_parquet(PROPS_PATH)
    td = props[
        (props.market == "player_anytime_td")
        & (props.side == "Yes")
        & props.position.isin(SKILL_POSITIONS)
        & props.player_id.notna()
    ].copy()

    idx = pd.read_parquet(EVENT_INDEX_PATH)[["event_id", "game_id", "season", "week", "home_team", "away_team"]]
    td = td.merge(idx, on="event_id", how="inner")
    td["p_raw"] = 1.0 / td["price"]
    return td


def compute_actual_scorers(seasons=ACTUAL_SCORER_SEASONS) -> pd.DataFrame:
    """One row per (game_id, player_id) that actually scored a rushing or
    receiving touchdown, restricted to QB/RB/WR/TE and the given seasons."""
    cols = ["season", "game_id", "touchdown", "td_player_id"]
    pbp = pd.read_parquet(PBP_PATH, columns=cols)
    pbp = pbp[pbp.season.isin(seasons) & (pbp.touchdown == 1) & pbp.td_player_id.notna()]

    rosters = pd.read_parquet(ROSTERS_PATH, columns=["season", "player_id", "position"])
    pos_lookup = rosters.drop_duplicates(subset=["season", "player_id"]).set_index(["season", "player_id"])["position"]

    scorers = pbp[["season", "game_id", "td_player_id"]].drop_duplicates()
    scorers["position"] = scorers.set_index(["season", "td_player_id"]).index.map(pos_lookup)
    scorers = scorers[scorers.position.isin(SKILL_POSITIONS)]
    return scorers.rename(columns={"td_player_id": "player_id"})[["season", "game_id", "player_id", "position"]]


def load_game_totals() -> pd.DataFrame:
    """game_id, total_line — nflverse's own closing total for the game."""
    sched = pd.read_parquet(SCHEDULES_PATH, columns=["game_id", "total_line"])
    return sched.dropna(subset=["total_line"])


def fit_expected_scorers_walkforward(per_game_actual: pd.DataFrame, game_totals: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward (train strictly-earlier seasons) OLS: actual_scorers ~
    total_line. Prints the fit per fold, returns game_id -> predicted
    expected_scorers for every game in an evaluable (non-first) season."""
    df = per_game_actual.merge(game_totals, on="game_id", how="inner")
    seasons = sorted(df.season.unique())

    print("\nExpected-scorers fit (actual_scorers ~ total_line), walk-forward by season:")
    preds = []
    for season in seasons[1:]:
        train = df[df.season < season]
        test = df[df.season == season]
        model = smf.ols("actual_scorers ~ total_line", data=train).fit()
        intercept, slope = model.params["Intercept"], model.params["total_line"]
        pred = model.predict(test)
        mae = float((pred - test["actual_scorers"]).abs().mean())
        print(f"  train<{season} (n={len(train)}): intercept={intercept:+.3f}  slope={slope:+.4f}  "
              f"R^2={model.rsquared:.3f}  test_mae={mae:.3f}")
        preds.append(pd.DataFrame({"game_id": test.game_id.values, "expected_scorers": pred.values}))

    if seasons:
        print(f"  (season {seasons[0]} excluded — no earlier season to train on)")

    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(columns=["game_id", "expected_scorers"])


def compute_margin_devig(td: pd.DataFrame, expected_scorers: pd.DataFrame) -> pd.DataFrame:
    """Adds expected_scorers, margin `m`, and p_devig = p_raw / m. Rows for
    games without a walk-forward expected_scorers prediction (the first
    season) are dropped."""
    td = td.merge(expected_scorers, on="game_id", how="inner")

    raw_sum = td.groupby(["game_id", "book"])["p_raw"].transform("sum")
    td["m"] = raw_sum / td["expected_scorers"]
    td["p_devig"] = td["p_raw"] / td["m"]
    return td


def report_overround(td: pd.DataFrame) -> None:
    print("=" * 70)
    print("PART 1 — Raw overround: sum of raw p(Yes) across all players, per (game, book)")
    print("=" * 70)
    per_game_book = td.groupby(["game_id", "book"])["p_raw"].sum()
    print(f"n = {len(per_game_book)} (game, book) pairs")
    print(per_game_book.describe(percentiles=[0.25, 0.5, 0.75]).to_string())


def report_devig(td: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PART 2 — Margin-based de-vig")
    print("=" * 70)

    print("\nMargin m — distribution across (game, book) markets:")
    m_per_market = td.groupby(["game_id", "book"])["m"].first()
    print(m_per_market.describe(percentiles=[0.25, 0.5, 0.75]).to_string())

    print("\nResulting summed probability after de-vig, per (game, book) "
          "(== that game's expected_scorers by construction, so this distribution "
          "is really the expected_scorers distribution, not a fixed constant like v1's 1.0):")
    devig_sum = td.groupby(["game_id", "book"])["p_devig"].sum()
    print(devig_sum.describe(percentiles=[0.25, 0.5, 0.75]).to_string())


def report_actual_scorers(scorers: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(f"PART 3 — Actual distinct QB/RB/WR/TE TD scorers per game, {ACTUAL_SCORER_SEASONS}")
    print("=" * 70)
    per_game = scorers.groupby(["season", "game_id"])["player_id"].nunique()
    print(f"n = {len(per_game)} games")
    print(f"mean = {per_game.mean():.3f}")
    print("\nFull distribution:")
    print(per_game.value_counts().sort_index().to_string())
    return per_game.rename("actual_scorers").reset_index()


def report_comparison(td: pd.DataFrame, per_game_actual: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PART 4 — De-vigged sum (part 2) vs actual distinct scorers (part 3)")
    print("=" * 70)
    print("Note: de-vigged sum is book-invariant by construction (always equals that "
          "game's expected_scorers, regardless of book) — 'by book' below reflects each "
          "book's game mix, not a book-specific de-vig effect.")

    devig_sum = td.groupby(["game_id", "book", "season"])["p_devig"].sum().reset_index()
    merged = devig_sum.merge(per_game_actual[["game_id", "actual_scorers"]], on="game_id", how="inner")

    print("\nBy season (mean across books):")
    print(merged.groupby("season")[["p_devig", "actual_scorers"]].mean().to_string())

    print("\nBy book (mean across seasons):")
    print(merged.groupby("book")[["p_devig", "actual_scorers"]].mean().sort_values("actual_scorers").to_string())


def load_scratch_lookup(seasons=ACTUAL_SCORER_SEASONS) -> dict:
    """{(season, week, team, normalized_name): offense_snaps} from
    snap_counts.parquet, for scratch detection by name (snap_counts only
    carries a PFR id, not the GSIS id used everywhere else)."""
    # No position filter here: snap_counts labels some props-side "RB"s (e.g.
    # blocking fullbacks) as "FB" instead — filtering to SKILL_POSITIONS would
    # exclude their real snap rows entirely and misclassify them as scratches
    # regardless of team/week/name match (verified: Kyle Juszczyk, Patrick
    # Ricard, Reggie Gilliam, C.J. Ham all do this). Participation doesn't
    # depend on which label snap_counts happens to use for the position.
    snaps = pd.read_parquet(SNAP_COUNTS_PATH)
    snaps = snaps[snaps.season.isin(seasons)].copy()
    snaps["norm_name"] = snaps["player"].map(normalize_name)
    snaps = snaps.groupby(["season", "week", "team", "norm_name"])["offense_snaps"].sum()
    return snaps.to_dict()


def load_player_teams(seasons=ACTUAL_SCORER_SEASONS) -> dict:
    """{(season, player_id): team} — one team per player per season (any
    week; used only to look up the snap-count row, not for correctness-
    critical logic, consistent with how team assignment is handled
    elsewhere in this project)."""
    rosters = pd.read_parquet(ROSTERS_PATH, columns=["season", "player_id", "team"])
    rosters = rosters[rosters.season.isin(seasons)]
    return rosters.drop_duplicates(subset=["season", "player_id"]).set_index(["season", "player_id"])["team"].to_dict()


def compute_is_scratch(td: pd.DataFrame) -> pd.Series:
    player_teams = load_player_teams()
    scratch_lookup = load_scratch_lookup()

    teams = td.set_index(["season", "player_id"]).index.map(player_teams)
    norm_names = td["player_name"].map(normalize_name)
    keys = list(zip(td["season"], td["week"], teams, norm_names))
    snaps = pd.Series(keys, index=td.index).map(scratch_lookup)
    return snaps.isna() | (snaps == 0)


def report_calibration(td: pd.DataFrame, actual_scorer_pairs: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PART 5 — Calibration: predicted de-vigged probability vs actual hit rate")
    print("=" * 70)

    scored = set(zip(actual_scorer_pairs.game_id, actual_scorer_pairs.player_id))

    is_scratch = compute_is_scratch(td)
    n_scratch = int(is_scratch.sum())
    print(f"\nProp rows that are scratches (no snap-count row, or offense_snaps==0): "
          f"{n_scratch} / {len(td)} ({n_scratch / len(td) * 100:.2f}%)")

    df = td[~is_scratch].copy()
    df["actual"] = list(zip(df.game_id, df.player_id))
    df["actual"] = df["actual"].isin(scored)

    def calibration_table(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        sub["decile"] = pd.qcut(sub["p_devig"], q=10, duplicates="drop")
        return sub.groupby("decile", observed=True).agg(
            n=("actual", "size"), predicted=("p_devig", "mean"), actual_hit_rate=("actual", "mean"))

    print("\n--- Pooled across books, deciles ---")
    print(calibration_table(df).to_string())

    for position in SKILL_POSITIONS:
        pos_df = df[df.position == position]
        print(f"\n--- {position} only, deciles (n={len(pos_df)}) ---")
        print(calibration_table(pos_df).to_string())


def main() -> None:
    td = load_td_props()
    print(f"Loaded {len(td)} anytime-TD Yes-side QB/RB/WR/TE prop rows")

    report_overround(td)

    scorers = compute_actual_scorers()
    per_game_actual = report_actual_scorers(scorers)

    game_totals = load_game_totals()
    expected_scorers = fit_expected_scorers_walkforward(per_game_actual, game_totals)

    n_before = td.game_id.nunique()
    td = compute_margin_devig(td, expected_scorers)
    n_after = td.game_id.nunique()
    print(f"\n(Games in the first season, {sorted(per_game_actual.season.unique())[0]}, have no walk-forward "
          f"expected_scorers prediction and are excluded from de-vig onward: {n_after}/{n_before} games retained)")

    report_devig(td)
    report_comparison(td, per_game_actual)
    report_calibration(td, scorers)


if __name__ == "__main__":
    main()
