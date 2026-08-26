"""Book disagreement in the anytime-TD props market: how much do books'
de-vigged probabilities disagree per player, and when one book sits far
from its peers, is IT or the peer median the sharper predictor?

Reuses src.eval.props_sanity's margin-based de-vig, walk-forward
expected-scorers fit, and snap_counts-based scratch detection wholesale —
see that module for why de-vig works this way (per-leg margin, not
Shin/proportional field normalization) and how scratches are identified
(snap_counts, not pbp participation).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.props_sanity import (
    EVENT_INDEX_PATH, PROPS_PATH, compute_actual_scorers, compute_is_scratch,
    compute_margin_devig, fit_expected_scorers_walkforward, load_game_totals,
    load_td_props,
)

OUTLIER_THRESHOLDS = (0.02, 0.04, 0.06)
MIN_BOOKS_FOR_OUTLIER = 3  # need >=2 "others" for a meaningful median-of-others


def build_devigged_props() -> pd.DataFrame:
    """One row per (game_id, player_id, book): p_devig plus `actual` (1 if
    that player scored in that game), scratches excluded."""
    td = load_td_props()
    scorers = compute_actual_scorers()
    per_game_actual = scorers.groupby(["season", "game_id"])["player_id"].nunique().rename(
        "actual_scorers").reset_index()
    game_totals = load_game_totals()
    expected_scorers = fit_expected_scorers_walkforward(per_game_actual, game_totals)

    td = compute_margin_devig(td, expected_scorers)
    is_scratch = compute_is_scratch(td)
    td = td[~is_scratch].copy()

    scored = set(zip(scorers.game_id, scorers.player_id))
    td["actual"] = list(zip(td.game_id, td.player_id))
    td["actual"] = td["actual"].isin(scored).astype(int)

    return td


def report_spread(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("PART 1 — Max-min spread of de-vigged probability across books, per (game_id, player_id)")
    print("=" * 70)
    grp = df.groupby(["game_id", "player_id"])["p_devig"]
    n_books = grp.size()
    spread = grp.max() - grp.min()

    print(f"n = {len(spread)} (game, player) pairs")
    print(f"  of which {int((n_books == 1).sum())} have only 1 book quoting (spread trivially 0)")

    print("\nFull distribution (all groups, including single-book):")
    print(spread.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string())

    multi = spread[n_books >= 2]
    print(f"\nDistribution restricted to groups with >=2 books (n={len(multi)}):")
    print(multi.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string())


def compute_outlier_table(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One row per (game_id, player_id, book) among groups with
    >=MIN_BOOKS_FOR_OUTLIER books quoting, where that book's p_devig
    differs from the leave-one-out median of the OTHER books by more than
    `threshold` (in probability units, e.g. 0.02 = 2 percentage points)."""
    rows = []
    for (game_id, player_id), g in df.groupby(["game_id", "player_id"]):
        if len(g) < MIN_BOOKS_FOR_OUTLIER:
            continue
        values = g["p_devig"].to_numpy()
        books = g["book"].to_numpy()
        prices = g["price"].to_numpy()
        p_raws = g["p_raw"].to_numpy()
        season = g["season"].iloc[0]
        actual = int(g["actual"].iloc[0])

        for i in range(len(values)):
            med_others = np.median(np.delete(values, i))
            diff = values[i] - med_others
            if abs(diff) > threshold:
                rows.append({
                    "game_id": game_id, "player_id": player_id, "book": books[i],
                    "season": season, "price": prices[i], "p_raw": p_raws[i],
                    "book_p": values[i], "median_others": med_others, "diff": diff,
                    "direction": "above" if diff > 0 else "below", "actual": actual,
                })
    return pd.DataFrame(rows)


def report_outlier_counts(df: pd.DataFrame, outlier_tables: dict) -> None:
    print("\n" + "=" * 70)
    print(f"PART 2 — Outlier counts (book differs from median-of-others by more than X), "
          f"min {MIN_BOOKS_FOR_OUTLIER} books/group")
    print("=" * 70)
    n_eligible_groups = int((df.groupby(["game_id", "player_id"]).size() >= MIN_BOOKS_FOR_OUTLIER).sum())
    n_eligible_rows = int(df.groupby(["game_id", "player_id"]).filter(
        lambda g: len(g) >= MIN_BOOKS_FOR_OUTLIER).shape[0])
    print(f"Eligible groups (>={MIN_BOOKS_FOR_OUTLIER} books): {n_eligible_groups}  "
          f"({n_eligible_rows} book-level rows)")
    for threshold in OUTLIER_THRESHOLDS:
        n = len(outlier_tables[threshold])
        print(f"  X={threshold * 100:.0f}pp: {n} outlier cases ({n / n_eligible_rows * 100:.2f}% of eligible rows)")


def report_predictive_comparison(outlier_tables: dict) -> None:
    print("\n" + "=" * 70)
    print("PART 3 — Does the outlier book or the peer median predict better?")
    print("=" * 70)

    for threshold in OUTLIER_THRESHOLDS:
        outliers = outlier_tables[threshold]
        print(f"\n--- X = {threshold * 100:.0f}pp (n={len(outliers)}) ---")
        if outliers.empty:
            print("  no outlier cases at this threshold")
            continue

        brier_outlier = ((outliers.book_p - outliers.actual) ** 2).mean()
        brier_median = ((outliers.median_others - outliers.actual) ** 2).mean()
        sharper = "outlier book" if brier_outlier < brier_median else "peer median"
        print(f"  Brier — outlier book: {brier_outlier:.4f}   peer median: {brier_median:.4f}   "
              f"({sharper} is sharper)")

        for direction in ("above", "below"):
            sub = outliers[outliers.direction == direction]
            if sub.empty:
                continue
            print(f"  outlier {direction} median (n={len(sub)}): "
                  f"mean_outlier_p={sub.book_p.mean():.3f}  mean_median_p={sub.median_others.mean():.3f}  "
                  f"actual_hit_rate={sub.actual.mean():.3f}")


def report_by_book(df: pd.DataFrame, outlier_tables: dict) -> None:
    print("\n" + "=" * 70)
    print("PART 4 — By book: who is systematically the sharp one, who is the soft one?")
    print("=" * 70)

    eligible = df.groupby(["game_id", "player_id"]).filter(lambda g: len(g) >= MIN_BOOKS_FOR_OUTLIER)
    total_by_book = eligible["book"].value_counts()

    for threshold in OUTLIER_THRESHOLDS:
        outliers = outlier_tables[threshold]
        print(f"\n--- X = {threshold * 100:.0f}pp ---")
        if outliers.empty:
            print("  no outlier cases at this threshold")
            continue

        rows = []
        for book in sorted(outliers.book.unique()):
            sub = outliers[outliers.book == book]
            n_total = int(total_by_book.get(book, 0))
            brier_outlier = ((sub.book_p - sub.actual) ** 2).mean()
            brier_median = ((sub.median_others - sub.actual) ** 2).mean()
            rows.append({
                "book": book, "n_outlier": len(sub), "n_eligible": n_total,
                "outlier_rate": len(sub) / n_total if n_total else float("nan"),
                "n_above": int((sub.direction == "above").sum()), "n_below": int((sub.direction == "below").sum()),
                "brier_book": brier_outlier, "brier_median": brier_median,
                "book_sharper": brier_outlier < brier_median,
            })
        table = pd.DataFrame(rows).set_index("book").sort_values("outlier_rate", ascending=False)
        print(table.to_string())


def report_bettable_side(outlier_tables: dict) -> None:
    """Below-median outliers: the book prices this player's TD as LESS
    likely than its peers, i.e. offers longer (more generous) odds than
    the market consensus implies — the side you'd actually back."""
    print("\n" + "=" * 70)
    print("PART 5 — Bettable side: below-median outliers, backed at the outlier book's own price")
    print("=" * 70)

    for threshold in OUTLIER_THRESHOLDS:
        below = outlier_tables[threshold]
        below = below[below.direction == "below"].copy()
        print(f"\n--- X = {threshold * 100:.0f}pp (n={len(below)}) ---")
        if below.empty:
            print("  no below-median outlier cases at this threshold")
            continue

        below["profit"] = below["actual"] * below["price"] - 1.0

        print(f"  POOLED: n={len(below)}  hit_rate={below.actual.mean():.3f}  "
              f"raw_implied_p(w/ vig)={below.p_raw.mean():.3f}  ROI={below.profit.mean() * 100:+.2f}%")

        print("\n  By book:")
        rows = []
        for book, sub in below.groupby("book"):
            rows.append({
                "book": book, "n": len(sub), "hit_rate": sub.actual.mean(),
                "raw_implied_p": sub.p_raw.mean(), "roi_pct": sub.profit.mean() * 100,
            })
        print(pd.DataFrame(rows).set_index("book").sort_values("n", ascending=False).to_string())

        print("\n  By season:")
        rows = []
        for season, sub in below.groupby("season"):
            rows.append({
                "season": season, "n": len(sub), "hit_rate": sub.actual.mean(),
                "raw_implied_p": sub.p_raw.mean(), "roi_pct": sub.profit.mean() * 100,
            })
        print(pd.DataFrame(rows).set_index("season").to_string())


def load_no_side_keys() -> set:
    """{(game_id, player_id, book)} for every anytime-TD 'No' quote ever
    offered — used to check which above-median outliers would have been
    layable via an actual No price rather than needing a synthetic lay."""
    props = pd.read_parquet(PROPS_PATH)
    no = props[
        (props.market == "player_anytime_td") & (props.side == "No") & props.player_id.notna()
    ].copy()
    idx = pd.read_parquet(EVENT_INDEX_PATH)[["event_id", "game_id"]]
    no = no.merge(idx, on="event_id", how="inner")
    return set(zip(no.game_id, no.player_id, no.book))


def report_layable_no_side(outlier_tables: dict) -> None:
    print("\n" + "=" * 70)
    print("PART 6 — Above-median outliers where a 'No' price existed (layable)")
    print("=" * 70)

    no_keys = load_no_side_keys()
    for threshold in OUTLIER_THRESHOLDS:
        above = outlier_tables[threshold]
        above = above[above.direction == "above"].copy()
        has_no = above.apply(lambda r: (r.game_id, r.player_id, r.book) in no_keys, axis=1)
        n_layable = int(has_no.sum())
        print(f"\n--- X = {threshold * 100:.0f}pp: {n_layable} / {len(above)} above-median outliers had a No price ---")
        if n_layable:
            print(above[has_no].groupby("book").size().sort_values(ascending=False).to_string())


def main() -> None:
    df = build_devigged_props()
    print(f"Non-scratch anytime-TD prop rows: {len(df)}")

    report_spread(df)

    outlier_tables = {t: compute_outlier_table(df, t) for t in OUTLIER_THRESHOLDS}
    report_outlier_counts(df, outlier_tables)
    report_predictive_comparison(outlier_tables)
    report_by_book(df, outlier_tables)
    report_bettable_side(outlier_tables)
    report_layable_no_side(outlier_tables)


if __name__ == "__main__":
    main()
