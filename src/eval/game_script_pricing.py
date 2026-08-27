"""Test whether the book already prices game script into the RB
rush-attempts line, before building any model on top of it.

If actual rush attempts respond MORE steeply to the spread (favourites
run more to protect a lead, run out the clock) than the book's own line
does, the book is under-adjusting for game script and the gap between
the two slopes is a candidate edge — worth checking before spending
effort modelling anything fancier on top of this market.

`favored_by` is the spread from the RB'S OWN TEAM's perspective, sign
flipped from this project's usual book convention so it reads naturally:
positive = that team favoured by that many points, negative = underdog.
(validate_featured.py's close_spread is home-perspective, book convention,
home favourite = negative; favored_by = -close_spread for the home team's
players, +close_spread for the away team's.)

Pure offline analysis — no network calls, only cached parquet files.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.eval.book_disagreement import (
    compute_actual_ou_stats, load_ou_props, select_primary_lines,
)
from src.eval.props_sanity import load_player_teams
from src.eval.validate_featured import (
    build_event_schedule_map, compute_close_spreads,
    compute_consensus_home_spreads, load_data,
)

MARKET = "player_rush_attempts"
POSITION = "RB"

SPREAD_BUCKET_EDGES = [-np.inf, -7, -3, 3, 7, np.inf]
SPREAD_BUCKET_LABELS = ["dog 7+", "dog 3-7", "pk to 3", "fav 3-7", "fav 7+"]


def load_team_spread() -> pd.DataFrame:
    """game_id, home_team, away_team, close_spread (home perspective, book
    convention: home favourite = negative)."""
    featured, schedules = load_data()
    event_map = build_event_schedule_map(featured, schedules)
    games = event_map.drop_duplicates(subset="game_id")
    consensus = compute_consensus_home_spreads(featured, event_map)
    close = compute_close_spreads(consensus, games)
    return close.merge(games[["game_id", "home_team", "away_team"]], on="game_id")


def _attach_favored_by(df: pd.DataFrame, team_spread: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(team_spread, on="game_id", how="inner")
    df["favored_by"] = np.where(df.team == df.home_team, -df.close_spread, df.close_spread)
    return df


def build_rb_line_dataset() -> pd.DataFrame:
    """One row per (game_id, player_id) with a rush-attempts line:
    book_line = median primary line across books, actual = actual rush
    attempts (0 if the RB has a line but no qualifying pbp rows),
    favored_by. Scratches already excluded (load_ou_props)."""
    ou = load_ou_props()
    rb = ou[(ou.market == MARKET) & (ou.position == POSITION)].copy()

    primary = select_primary_lines(rb)
    book_lines = primary.drop_duplicates(["game_id", "player_id", "book"])
    consensus_line = book_lines.groupby(["game_id", "player_id"]).agg(
        book_line=("line", "median"), n_books=("book", "nunique")).reset_index()

    season_lookup = rb.drop_duplicates(["game_id", "player_id"])[["game_id", "player_id", "season"]]
    consensus_line = consensus_line.merge(season_lookup, on=["game_id", "player_id"], how="left")

    actual = compute_actual_ou_stats()[MARKET]
    df = consensus_line.merge(actual, on=["game_id", "player_id"], how="left")
    df["actual"] = df["actual"].fillna(0.0)

    team_lookup = load_player_teams()
    df["team"] = df.set_index(["season", "player_id"]).index.map(team_lookup)

    df = _attach_favored_by(df, load_team_spread())
    return df.dropna(subset=["book_line", "actual", "favored_by", "team"])


def build_rb_quote_dataset() -> pd.DataFrame:
    """One row per (game_id, player_id, book) Under quote (all lines, not
    just the primary/consensus one) — for the 'raw, at book prices'
    hit-rate/ROI check in part 5."""
    ou = load_ou_props()
    rb = ou[(ou.market == MARKET) & (ou.position == POSITION) & (ou.side == "Under")].copy()
    rb = rb.drop(columns=["home_team", "away_team"])

    actual = compute_actual_ou_stats()[MARKET]
    df = rb.merge(actual, on=["game_id", "player_id"], how="left")
    df["actual"] = df["actual"].fillna(0.0)

    team_lookup = load_player_teams()
    df["team"] = df.set_index(["season", "player_id"]).index.map(team_lookup)

    df = _attach_favored_by(df, load_team_spread())
    df = df.dropna(subset=["line", "actual", "favored_by", "team"])

    df["outcome"] = np.where(df.actual == df.line, "push", np.where(df.actual < df.line, "win", "loss"))
    df["profit"] = np.where(df.outcome == "push", np.nan, np.where(df.outcome == "win", df["price"] - 1.0, -1.0))
    return df


def fit_slope(df: pd.DataFrame, target_col: str, label: str) -> "smf.ols":
    model = smf.ols(f"{target_col} ~ favored_by", data=df).fit()
    slope = model.params["favored_by"]
    print(f"  {label}: n={len(df)}  intercept={model.params['Intercept']:.3f}  "
          f"slope={slope:+.4f} attempts/point  se={model.bse['favored_by']:.4f}  "
          f"p={model.pvalues['favored_by']:.4g}  R^2={model.rsquared:.3f}")
    return model


def part1_book_line_slope(df: pd.DataFrame):
    print("=" * 70)
    print("PART 1 — Book's rush-attempts LINE regressed on favored_by (consensus line, one row per RB-game)")
    print("=" * 70)
    return fit_slope(df, "book_line", "book line ~ favored_by")


def part2_actual_slope(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("PART 2 — ACTUAL rush attempts regressed on the same favored_by (same RB-game sample)")
    print("=" * 70)
    return fit_slope(df, "actual", "actual attempts ~ favored_by")


def part3_compare(book_model, actual_model) -> None:
    print("\n" + "=" * 70)
    print("PART 3 — Compare: does the book under-adjust for game script?")
    print("=" * 70)
    book_slope = book_model.params["favored_by"]
    actual_slope = actual_model.params["favored_by"]
    gap = actual_slope - book_slope
    print(f"  book slope   = {book_slope:+.4f} attempts/point")
    print(f"  actual slope = {actual_slope:+.4f} attempts/point")
    print(f"  gap (actual - book) = {gap:+.4f} attempts/point")
    if abs(book_slope) < abs(actual_slope):
        print(f"  -> book slope is SHALLOWER than reality: book under-adjusts for game script. "
              f"Candidate edge ~ {gap:+.4f} attempts per point of spread.")
    else:
        print("  -> book slope is NOT shallower than reality — no evidence of under-adjustment here.")


def part4_buckets(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PART 4 — Regressions and actual-minus-line residual, by spread bucket")
    print("=" * 70)
    df = df.copy()
    df["bucket"] = pd.cut(df["favored_by"], bins=SPREAD_BUCKET_EDGES, labels=SPREAD_BUCKET_LABELS)

    for label in SPREAD_BUCKET_LABELS:
        sub = df[df.bucket == label]
        print(f"\n--- {label} (n={len(sub)}) ---")
        if len(sub) < 5:
            print("  too few observations to regress")
            continue
        fit_slope(sub, "book_line", "book line ~ favored_by")
        fit_slope(sub, "actual", "actual attempts ~ favored_by")
        residual = sub["actual"] - sub["book_line"]
        print(f"  actual-minus-line residual: mean={residual.mean():+.3f}  n={len(sub)}")


def part5_under_by_bucket(quotes: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PART 5 — Raw Under hit rate and ROI at book prices, by spread bucket "
          "(every book/line quote, not just the consensus line)")
    print("=" * 70)
    quotes = quotes.copy()
    quotes["bucket"] = pd.cut(quotes["favored_by"], bins=SPREAD_BUCKET_EDGES, labels=SPREAD_BUCKET_LABELS)

    for label in SPREAD_BUCKET_LABELS:
        sub = quotes[quotes.bucket == label]
        decided = sub[sub.outcome != "push"]
        n_push = len(sub) - len(decided)
        if decided.empty:
            print(f"  {label:8}: no decided bets")
            continue
        hit_rate = (decided.outcome == "win").mean()
        roi = decided["profit"].mean() * 100
        print(f"  {label:8}: n={len(sub):5d}  n_push={n_push:3d}  hit_rate={hit_rate:.3f}  ROI={roi:+.2f}%")


def main() -> None:
    df = build_rb_line_dataset()
    print(f"RB rush-attempts lines with a matched spread: {len(df)}\n")

    book_model = part1_book_line_slope(df)
    actual_model = part2_actual_slope(df)
    part3_compare(book_model, actual_model)
    part4_buckets(df)

    quotes = build_rb_quote_dataset()
    part5_under_by_bucket(quotes)


if __name__ == "__main__":
    main()
