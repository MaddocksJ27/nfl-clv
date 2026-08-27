"""Test whether books set O/U lines near the conditional MEAN of a
right-skewed outcome distribution rather than the conditional MEDIAN. If
so, Under is mechanically more likely to hit than 50% for reasons that
have nothing to do with any real forecasting skill — a candidate
explanation for the Under-side ROI edge found in book_disagreement.py
and game_script_pricing.py.

Deduplicated to one row per (game_id, player_id, market): consensus_line
is the median primary line across books (same definition used in
game_script_pricing.py), consensus_price_under is the median Under price
across books at their own primary line — a single representative bet per
player-game, rather than one row per book.

Pure offline analysis — no network calls, only cached parquet files.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.eval.book_disagreement import (
    OU_MARKETS, bootstrap_roi, compute_actual_ou_stats, load_ou_props,
    select_primary_lines,
)

MAX_EXACT_BUCKETS = 20  # markets with <= this many distinct consensus lines get exact-value buckets
N_QUANTILE_BUCKETS = 8  # otherwise, quantile bins on the consensus line


def build_consensus_dataset(ou: pd.DataFrame, market: str) -> pd.DataFrame:
    """One row per (game_id, player_id): consensus_line, consensus_price_under,
    n_books, actual. Scratches already excluded (load_ou_props)."""
    sub = ou[ou.market == market].copy()
    primary = select_primary_lines(sub)
    book_rows = primary.drop_duplicates(["game_id", "player_id", "book"])

    consensus_line = book_rows.groupby(["game_id", "player_id"]).agg(
        consensus_line=("line", "median"), n_books=("book", "nunique")).reset_index()

    under_prices = primary[primary.side == "Under"]
    consensus_price = under_prices.groupby(["game_id", "player_id"])["price"].median().rename(
        "consensus_price_under").reset_index()

    df = consensus_line.merge(consensus_price, on=["game_id", "player_id"], how="left")

    actual = compute_actual_ou_stats()[market]
    df = df.merge(actual, on=["game_id", "player_id"], how="left")
    df["actual"] = df["actual"].fillna(0.0)
    return df


def make_buckets(df: pd.DataFrame) -> pd.Series:
    """Exact consensus-line value for markets with few distinct lines
    (receptions, rush attempts); quantile bins of the consensus line for
    the continuous yardage markets, where hundreds of distinct lines make
    exact-value buckets mostly singletons."""
    n_unique = df["consensus_line"].nunique()
    if n_unique <= MAX_EXACT_BUCKETS:
        return df["consensus_line"]
    return pd.qcut(df["consensus_line"], q=N_QUANTILE_BUCKETS, duplicates="drop")


def report_distribution(df: pd.DataFrame, market: str) -> None:
    actual = df["actual"]
    skewness = scipy_stats.skew(actual)
    print(f"  n={len(df)}  mean={actual.mean():.3f}  median={actual.median():.3f}  skew={skewness:+.3f}")


def report_line_vs_mean_median(df: pd.DataFrame, market: str) -> None:
    df = df.copy()
    df["bucket"] = make_buckets(df)
    table = df.groupby("bucket", observed=True).agg(
        n=("actual", "size"), consensus_line=("consensus_line", "mean"),
        actual_mean=("actual", "mean"), actual_median=("actual", "median"),
    )
    table["line_minus_mean"] = table["consensus_line"] - table["actual_mean"]
    table["line_minus_median"] = table["consensus_line"] - table["actual_median"]
    print(table.round(3).to_string())


def report_skew_implied_p_under(df: pd.DataFrame, market: str) -> None:
    actual = df["actual"]
    mean = actual.mean()
    # Mechanical consequence of right skew IF the book set the line at the
    # distribution's mean: more than half the mass sits below the mean.
    p_under_from_skew = (actual < mean).mean()

    decided = df[df["actual"] != df["consensus_line"]]
    observed_hit_rate = (decided["actual"] < decided["consensus_line"]).mean()

    print(f"  P(actual < empirical mean) [skew-implied, if line==mean] = {p_under_from_skew:.3f}")
    print(f"  observed P(actual < book's consensus line)               = {observed_hit_rate:.3f}")
    print(f"  gap (observed - skew-implied) = {observed_hit_rate - p_under_from_skew:+.3f}")


def report_bucketed_roi(df: pd.DataFrame, market: str) -> None:
    df = df.dropna(subset=["consensus_price_under"]).copy()
    df["outcome"] = np.where(df.actual == df.consensus_line, "push",
                              np.where(df.actual < df.consensus_line, "win", "loss"))
    df["profit"] = np.where(df.outcome == "push", np.nan,
                             np.where(df.outcome == "win", df["consensus_price_under"] - 1.0, -1.0))
    df["bucket"] = make_buckets(df)
    print(f"  ({len(df)} player-games with a consensus Under price; deduplicated sample)")

    rows = []
    for bucket, sub in df.groupby("bucket", observed=True):
        decided = sub[sub.outcome != "push"]
        if decided.empty:
            continue
        profits = decided["profit"].to_numpy()
        hit_rate = (decided.outcome == "win").mean()
        roi = profits.mean() * 100
        p5 = np.percentile(bootstrap_roi(profits), 5) * 100 if len(profits) > 1 else float("nan")
        rows.append({"bucket": str(bucket), "n": len(sub), "n_decided": len(decided),
                      "hit_rate": hit_rate, "roi_pct": roi, "boot_5th_pctile": p5})
    print(pd.DataFrame(rows).set_index("bucket").round(3).to_string())


def main() -> None:
    ou = load_ou_props()
    print(f"Non-scratch O/U prop rows across 5 markets: {len(ou)}\n")

    for market in OU_MARKETS:
        df = build_consensus_dataset(ou, market)
        print("=" * 70)
        print(f"{market}  (deduplicated: n={len(df)} game-player pairs)")
        print("=" * 70)

        print("\n1. Distribution of actual outcomes:")
        report_distribution(df, market)

        print("\n2. Book consensus line vs conditional mean/median, by line bucket:")
        report_line_vs_mean_median(df, market)

        print("\n3. P(Under) implied by skewness alone vs observed:")
        report_skew_implied_p_under(df, market)

        print("\n4. Bucketed Under ROI (deduplicated, bootstrap 5th percentile):")
        report_bucketed_roi(df, market)
        print()


if __name__ == "__main__":
    main()
