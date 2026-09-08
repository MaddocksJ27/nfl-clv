"""By-week calibration and MAE check for player_receptions and
player_rush_attempts closing lines: are early-season lines genuinely
softer (higher MAE against actuals, worse calibration) than lines later
in the season, once players/teams are more of a known quantity?

Uses each book's PRIMARY line only (see book_disagreement.py's
select_primary_lines) — alt lines would otherwise inflate MAE/hit-rate
noise without reflecting the market's actual central estimate. Every
book's own quote counts as one observation (not deduplicated to a single
consensus line), matching "the market's calibration and hit rate" as
actually observed, one book at a time.

Pure offline analysis — no network calls, only cached parquet files.
"""

import numpy as np
import pandas as pd

from src.eval.book_disagreement import (
    compute_actual_ou_stats, load_ou_props, select_primary_lines,
)

MARKETS = ("player_receptions", "player_rush_attempts")

WEEK_BUCKET_EDGES = [0.5, 4.5, 9.5, 14.5, 18.5]
WEEK_BUCKET_LABELS = ["1-4", "5-9", "10-14", "15-18"]


def build_market_dataset(ou: pd.DataFrame, market: str) -> pd.DataFrame:
    """One row per (game_id, player_id, book, side), primary lines only,
    with actual outcome and win/loss/push joined in. Regular season only
    (week 1-18); anything outside that range is dropped."""
    sub = ou[ou.market == market].copy()
    primary = select_primary_lines(sub)

    actual = compute_actual_ou_stats()[market]
    df = primary.merge(actual, on=["game_id", "player_id"], how="left")
    df["actual"] = df["actual"].fillna(0.0)

    n_before = len(df)
    df = df[df.week.between(1, 18)].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  ({n_dropped} rows outside week 1-18 dropped)")
    df["week_bucket"] = pd.cut(df["week"], bins=WEEK_BUCKET_EDGES, labels=WEEK_BUCKET_LABELS)

    df["outcome"] = np.select(
        [df.actual == df.line,
         (df.side == "Over") & (df.actual > df.line),
         (df.side == "Under") & (df.actual < df.line)],
        ["push", "win", "win"], default="loss")
    return df


def report_calibration_by_week(df: pd.DataFrame, market: str) -> None:
    print("=" * 70)
    print(f"{market} — calibration/hit-rate by week bucket (primary lines, all books)")
    print("=" * 70)

    for side in ("Over", "Under"):
        print(f"\n--- {side} ---")
        rows = []
        for bucket in WEEK_BUCKET_LABELS:
            sub = df[(df.week_bucket == bucket) & (df.side == side)]
            decided = sub[sub.outcome != "push"]
            if decided.empty:
                continue
            rows.append({
                "week": bucket, "n": len(sub), "n_push": len(sub) - len(decided),
                "mean_raw_implied_p": decided.p_raw.mean(),
                "hit_rate": (decided.outcome == "win").mean(),
            })
        print(pd.DataFrame(rows).set_index("week").round(4).to_string())


def report_mae_by_week(df: pd.DataFrame, market: str) -> None:
    print("\n--- MAE of line vs actual outcome, by week bucket ---")
    lines = df.drop_duplicates(["game_id", "player_id", "book"])[
        ["game_id", "player_id", "book", "week_bucket", "line", "actual"]].copy()
    lines["abs_error"] = (lines["line"] - lines["actual"]).abs()

    rows = []
    for bucket in WEEK_BUCKET_LABELS:
        sub = lines[lines.week_bucket == bucket]
        if sub.empty:
            continue
        rows.append({"week": bucket, "n": len(sub), "mae": sub.abs_error.mean(),
                      "median_ae": sub.abs_error.median()})
    print(pd.DataFrame(rows).set_index("week").round(4).to_string())


def main() -> None:
    ou = load_ou_props()
    print(f"Non-scratch O/U prop rows across 5 markets: {len(ou)}\n")

    for market in MARKETS:
        df = build_market_dataset(ou, market)
        report_calibration_by_week(df, market)
        report_mae_by_week(df, market)
        print()


if __name__ == "__main__":
    main()
