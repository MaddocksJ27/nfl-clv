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
    ACTUAL_SCORER_SEASONS, EVENT_INDEX_PATH, PBP_PATH, PROPS_PATH,
    SKILL_POSITIONS, compute_actual_scorers, compute_is_scratch,
    compute_margin_devig, fit_expected_scorers_walkforward, load_game_totals,
    load_td_props,
)

OUTLIER_THRESHOLDS = (0.02, 0.04, 0.06)
MIN_BOOKS_FOR_OUTLIER = 3  # need >=2 "others" for a meaningful median-of-others

OU_MARKETS = (
    "player_receptions", "player_rush_attempts", "player_rush_yds",
    "player_reception_yds", "player_pass_yds",
)


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


## ---------------------------------------------------------------------
## Over/Under markets: player_receptions, player_rush_attempts,
## player_rush_yds, player_reception_yds, player_pass_yds.
##
## These are two-sided (Over/Under) rather than a Yes/No field, and books
## also disagree on the LINE itself, not just the price at a shared line.
## Books frequently offer several alternate lines for the same player —
## ~9% of (market, game, player, book) combos have more than one distinct
## line. To keep "line disagreement between books" separate from "how
## many alt lines a book happens to list", most of what follows works off
## each book's PRIMARY line: among lines where that book quotes BOTH
## Over and Under, the one closest to the across-book median line for
## that player/game. Outlier and ROI analysis go one step further and
## restrict to (game, player, line) triples where >=3 books quote the
## IDENTICAL line — comparing de-vigged probabilities only makes sense
## when books are pricing the same bet.
## ---------------------------------------------------------------------


def load_ou_props() -> pd.DataFrame:
    """All five O/U markets, Over/Under sides, QB/RB/WR/TE, with
    game_id/season/week joined in and p_raw = 1/price. Scratches
    (via snap_counts, same definition as the anytime-TD market) are
    excluded here so they never enter any downstream O/U analysis."""
    props = pd.read_parquet(PROPS_PATH)
    ou = props[
        props.market.isin(OU_MARKETS)
        & props.side.isin(["Over", "Under"])
        & props.position.isin(SKILL_POSITIONS)
        & props.player_id.notna()
    ].copy()

    idx = pd.read_parquet(EVENT_INDEX_PATH)[["event_id", "game_id", "season", "week", "home_team", "away_team"]]
    ou = ou.merge(idx, on="event_id", how="inner")
    ou["p_raw"] = 1.0 / ou["price"]

    ou = ou[~compute_is_scratch(ou)].reset_index(drop=True)
    return ou


def report_ou_coverage(ou: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PART 7 — Coverage: five Over/Under markets (all lines, all books, non-scratch)")
    print("=" * 70)
    for market in OU_MARKETS:
        sub = ou[ou.market == market]
        n_pairs = sub.groupby(["game_id", "player_id"]).ngroups
        print(f"\n--- {market} ---")
        print(f"  overall: rows={len(sub)}  (game,player) pairs={n_pairs}  books={sub.book.nunique()}")
        rows = []
        for season, s in sub.groupby("season"):
            rows.append({
                "season": season, "n_rows": len(s),
                "n_game_player": s.groupby(["game_id", "player_id"]).ngroups,
                "n_books": s.book.nunique(),
            })
        print(pd.DataFrame(rows).set_index("season").to_string())


def select_primary_lines(ou: pd.DataFrame) -> pd.DataFrame:
    """One row per (market, game_id, player_id, book, side): each book's
    main line only. See module docstring for why (alt lines)."""
    key_cols = ["market", "game_id", "player_id", "book", "line"]
    ou = ou.copy()
    ou["_key"] = list(zip(*(ou[c] for c in key_cols)))

    sides = ou.groupby("_key")["side"].apply(lambda s: frozenset(s))
    twoway_keys = set(sides[sides == frozenset({"Over", "Under"})].index)
    twoway = ou[ou["_key"].isin(twoway_keys)]

    book_lines = twoway.drop_duplicates("_key")
    median_line = book_lines.groupby(["market", "game_id", "player_id"])["line"].median().rename("median_line")
    book_lines = book_lines.merge(median_line, on=["market", "game_id", "player_id"])
    book_lines["dist"] = (book_lines["line"] - book_lines["median_line"]).abs()
    primary = book_lines.sort_values("dist").drop_duplicates(
        ["market", "game_id", "player_id", "book"], keep="first")

    result = twoway[twoway["_key"].isin(set(primary["_key"]))]
    return result.drop(columns="_key")


def report_line_disagreement(primary: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PART 8 — Line disagreement: max-min LINE across books, per (game, player), primary lines only")
    print("=" * 70)
    book_lines = primary.drop_duplicates(["market", "game_id", "player_id", "book"])
    for market in OU_MARKETS:
        sub = book_lines[book_lines.market == market]
        grp = sub.groupby(["game_id", "player_id"])["line"]
        n_books = grp.size()
        spread = grp.max() - grp.min()
        multi = spread[n_books >= 2]
        print(f"\n--- {market} (n={len(multi)} pairs with >=2 books) ---")
        if multi.empty:
            print("  no pairs with >=2 books")
            continue
        pct_disagree = (multi > 0).mean() * 100
        print(f"  {pct_disagree:.1f}% of pairs show ANY line disagreement across books")
        print(multi.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_string())


def compute_same_line_wide(ou: pd.DataFrame) -> pd.DataFrame:
    """One row per (market, game_id, player_id, line, book, season),
    restricted to (market, game_id, player_id, line) triples where
    >=MIN_BOOKS_FOR_OUTLIER books quote that EXACT line (both sides).
    Columns price_Over/price_Under/p_raw_Over/p_raw_Under plus
    p_over_devig (proportional de-vig: p_raw_Over / (p_raw_Over +
    p_raw_Under))."""
    key_cols = ["market", "game_id", "player_id", "book", "line"]
    ou = ou.copy()
    ou["_bkey"] = list(zip(*(ou[c] for c in key_cols)))
    sides = ou.groupby("_bkey")["side"].apply(lambda s: frozenset(s))
    twoway_bkeys = set(sides[sides == frozenset({"Over", "Under"})].index)
    two_way = ou[ou["_bkey"].isin(twoway_bkeys)]

    wide = two_way.pivot_table(
        index=["market", "game_id", "player_id", "line", "book", "season"],
        columns="side", values=["price", "p_raw"], aggfunc="first",
    )
    wide.columns = [f"{val}_{side}" for val, side in wide.columns]
    wide = wide.reset_index()
    wide["p_over_devig"] = wide["p_raw_Over"] / (wide["p_raw_Over"] + wide["p_raw_Under"])

    n_books = wide.groupby(["market", "game_id", "player_id", "line"])["book"].transform("nunique")
    return wide[n_books >= MIN_BOOKS_FOR_OUTLIER].reset_index(drop=True)


def report_same_line_spread(same_line: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print(f"PART 9 — Same-line proportional de-vig spread (Over probability), "
          f"restricted to (game,player,line) triples with >={MIN_BOOKS_FOR_OUTLIER} books on the identical line")
    print("=" * 70)
    for market in OU_MARKETS:
        sub = same_line[same_line.market == market]
        grp = sub.groupby(["game_id", "player_id", "line"])["p_over_devig"]
        n_triples = grp.ngroups
        print(f"\n--- {market}: {n_triples} surviving (game,player,line) triples ---")
        if n_triples == 0:
            print("  none survive — same-line sample is empty for this market")
            continue
        if n_triples < 30:
            print(f"  WARNING: only {n_triples} triples — same-line sample is tiny, treat with caution")
        spread = grp.max() - grp.min()
        print(spread.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_string())


def compute_actual_ou_stats(seasons=ACTUAL_SCORER_SEASONS) -> dict:
    """{market: DataFrame[game_id, player_id, actual]} ground truth from
    play-by-play. Two-point plays excluded (same convention as the rest
    of this project). A (game_id, player_id) with no qualifying pbp rows
    at all (e.g. zero receptions) simply won't appear here — callers must
    left-join and fillna(0), not treat a missing row as unknown."""
    cols = ["season", "game_id", "complete_pass", "receiving_yards", "receiver_player_id",
            "rush_attempt", "rushing_yards", "rusher_player_id",
            "pass_attempt", "passing_yards", "passer_player_id", "two_point_attempt"]
    pbp = pd.read_parquet(PBP_PATH, columns=cols)
    pbp = pbp[pbp.season.isin(seasons) & (pbp.two_point_attempt != 1)]

    rec = pbp[(pbp.complete_pass == 1) & pbp.receiver_player_id.notna()]
    receptions = rec.groupby(["game_id", "receiver_player_id"]).size().rename("actual").reset_index()
    receptions = receptions.rename(columns={"receiver_player_id": "player_id"})
    reception_yds = rec.groupby(["game_id", "receiver_player_id"])["receiving_yards"].sum().rename(
        "actual").reset_index().rename(columns={"receiver_player_id": "player_id"})

    rush = pbp[(pbp.rush_attempt == 1) & pbp.rusher_player_id.notna()]
    rush_attempts = rush.groupby(["game_id", "rusher_player_id"]).size().rename("actual").reset_index()
    rush_attempts = rush_attempts.rename(columns={"rusher_player_id": "player_id"})
    rush_yds = rush.groupby(["game_id", "rusher_player_id"])["rushing_yards"].sum().rename(
        "actual").reset_index().rename(columns={"rusher_player_id": "player_id"})

    passed = pbp[(pbp.pass_attempt == 1) & pbp.passer_player_id.notna()]
    pass_yds = passed.groupby(["game_id", "passer_player_id"])["passing_yards"].sum().rename(
        "actual").reset_index().rename(columns={"passer_player_id": "player_id"})

    return {
        "player_receptions": receptions,
        "player_reception_yds": reception_yds,
        "player_rush_attempts": rush_attempts,
        "player_rush_yds": rush_yds,
        "player_pass_yds": pass_yds,
    }


def compute_ou_outlier_table(same_line: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One row per (market, game_id, player_id, line, book) where that
    book's proportionally-de-vigged Over probability differs from the
    leave-one-out median of its peers (on the same line) by more than
    `threshold`. bettable_side='Over' when the book's Over price is the
    generous one (p_over_devig below peer median); bettable_side='Under'
    when the book's Under price is the generous one — which is exactly
    the case where p_over_devig sits ABOVE the peer median, since
    p_under_devig = 1 - p_over_devig and the median is equivariant under
    that reflection (median(1-x) = 1-median(x)), so no separate Under-
    side pass is needed."""
    rows = []
    for (market, game_id, player_id, line), g in same_line.groupby(["market", "game_id", "player_id", "line"]):
        values = g["p_over_devig"].to_numpy()
        books = g["book"].to_numpy()
        season = g["season"].iloc[0]
        price_over = g["price_Over"].to_numpy()
        praw_over = g["p_raw_Over"].to_numpy()
        price_under = g["price_Under"].to_numpy()
        praw_under = g["p_raw_Under"].to_numpy()

        for i in range(len(values)):
            med_others = np.median(np.delete(values, i))
            diff = values[i] - med_others
            if abs(diff) <= threshold:
                continue
            if diff < 0:
                side, price, p_raw = "Over", price_over[i], praw_over[i]
            else:
                side, price, p_raw = "Under", price_under[i], praw_under[i]
            rows.append({
                "market": market, "game_id": game_id, "player_id": player_id, "line": line,
                "book": books[i], "season": season, "bettable_side": side,
                "price": price, "p_raw": p_raw,
            })
    return pd.DataFrame(rows)


def attach_ou_outcomes(outliers: pd.DataFrame, actual_stats: dict) -> pd.DataFrame:
    """Joins actual stat value, tags each row win/loss/push (push = actual
    exactly equals the line — a void, excluded from hit-rate/ROI)."""
    parts = []
    for market, sub in outliers.groupby("market"):
        merged = sub.merge(actual_stats[market], on=["game_id", "player_id"], how="left")
        merged["actual"] = merged["actual"].fillna(0.0)
        parts.append(merged)
    df = pd.concat(parts, ignore_index=True) if parts else outliers.assign(actual=pd.Series(dtype=float))

    conditions = [
        df["actual"] == df["line"],
        (df["bettable_side"] == "Over") & (df["actual"] > df["line"]),
        (df["bettable_side"] == "Under") & (df["actual"] < df["line"]),
    ]
    df["outcome"] = np.select(conditions, ["push", "win", "win"], default="loss")
    df["profit"] = np.where(df.outcome == "push", np.nan,
                             np.where(df.outcome == "win", df["price"] - 1.0, -1.0))
    df["hit"] = np.where(df.outcome == "push", np.nan, (df.outcome == "win").astype(float))
    return df


def report_ou_outlier_counts(same_line: pd.DataFrame, ou_outlier_tables: dict) -> None:
    print("\n" + "=" * 70)
    print(f"PART 10a — O/U outlier counts, min {MIN_BOOKS_FOR_OUTLIER} books on the identical line")
    print("=" * 70)
    n_eligible_rows = len(same_line)
    print(f"Eligible book-level rows (same line, >={MIN_BOOKS_FOR_OUTLIER} books): {n_eligible_rows}")
    for threshold in OUTLIER_THRESHOLDS:
        n = len(ou_outlier_tables[threshold])
        pct = n / n_eligible_rows * 100 if n_eligible_rows else float("nan")
        print(f"  X={threshold * 100:.0f}pp: {n} outlier cases ({pct:.2f}% of eligible rows)")


def _summarize(df: pd.DataFrame) -> dict:
    decided = df[df.outcome != "push"]
    return {
        "n": len(df), "n_push": int((df.outcome == "push").sum()),
        "n_decided": len(decided),
        "hit_rate": decided["hit"].mean() if len(decided) else float("nan"),
        "raw_implied_p": decided["p_raw"].mean() if len(decided) else float("nan"),
        "roi_pct": decided["profit"].mean() * 100 if len(decided) else float("nan"),
    }


def report_ou_bettable_side(outcomes: dict) -> None:
    """outcomes: {threshold: DataFrame} — output of attach_ou_outcomes,
    already restricted to outlier cases (both directions bettable)."""
    print("\n" + "=" * 70)
    print("PART 10b — O/U outliers, both directions bettable: hit rate vs raw (vig-inclusive) "
          "implied probability and ROI at the outlier book's own price. Pushes excluded from "
          "hit rate/ROI, reported separately.")
    print("=" * 70)

    for threshold in OUTLIER_THRESHOLDS:
        df = outcomes[threshold]
        print(f"\n--- X = {threshold * 100:.0f}pp (n={len(df)}) ---")
        if df.empty:
            print("  no outlier cases at this threshold")
            continue

        pooled = _summarize(df)
        print(f"  POOLED: n={pooled['n']}  n_push={pooled['n_push']}  n_decided={pooled['n_decided']}  "
              f"hit_rate={pooled['hit_rate']:.3f}  raw_implied_p={pooled['raw_implied_p']:.3f}  "
              f"ROI={pooled['roi_pct']:+.2f}%")

        for label, key in (("market", "market"), ("book", "book"), ("season", "season"), ("side", "bettable_side")):
            print(f"\n  By {label}:")
            rows = []
            for val, sub in df.groupby(key):
                s = _summarize(sub)
                rows.append({label: val, **s})
            table = pd.DataFrame(rows).set_index(label).sort_values("n", ascending=False)
            print(table.to_string())


def report_ou_pushes_and_scratches(ou: pd.DataFrame, outcomes: dict) -> None:
    print("\n" + "=" * 70)
    print("PART 11 — Push counts per market (among outlier cases) and scratch exclusion recap")
    print("=" * 70)
    print("\nScratch exclusion uses the same snap_counts-based definition as the anytime-TD "
          "market (see props_sanity.py); it was already applied when loading O/U props, "
          "before any of the above.")

    for threshold in OUTLIER_THRESHOLDS:
        df = outcomes[threshold]
        print(f"\n--- X = {threshold * 100:.0f}pp ---")
        if df.empty:
            print("  no outlier cases at this threshold")
            continue
        rows = []
        for market, sub in df.groupby("market"):
            rows.append({
                "market": market, "n": len(sub), "n_push": int((sub.outcome == "push").sum()),
                "push_rate": (sub.outcome == "push").mean(),
            })
        print(pd.DataFrame(rows).set_index("market").to_string())


def run_ou_markets() -> dict:
    ou = load_ou_props()
    print(f"\nNon-scratch O/U prop rows across 5 markets: {len(ou)}")

    report_ou_coverage(ou)

    primary = select_primary_lines(ou)
    report_line_disagreement(primary)

    same_line = compute_same_line_wide(ou)
    report_same_line_spread(same_line)

    ou_outlier_tables = {t: compute_ou_outlier_table(same_line, t) for t in OUTLIER_THRESHOLDS}
    report_ou_outlier_counts(same_line, ou_outlier_tables)

    actual_stats = compute_actual_ou_stats()
    outcomes = {t: attach_ou_outcomes(ou_outlier_tables[t], actual_stats) for t in OUTLIER_THRESHOLDS}
    report_ou_bettable_side(outcomes)
    report_ou_pushes_and_scratches(ou, outcomes)

    return {"ou": ou, "same_line": same_line, "actual_stats": actual_stats, "outcomes": outcomes}


## ---------------------------------------------------------------------
## Under/Over robustness deep-dive: is the Under-side edge found above
## real, or does it evaporate under a season/market/bootstrap breakdown?
## ---------------------------------------------------------------------


def bootstrap_roi(profits: np.ndarray, n_resamples: int = 10_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(profits)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = profits[idx].mean()
    return means


def report_side_robustness(outcomes: dict, side: str) -> dict:
    """Season and market breakdown of ROI for one bettable side's outlier
    cases, at each threshold. Returns {threshold: decided-profit array}
    for the bootstrap step."""
    print(f"\n--- {side}-side outliers: by season ---")
    profits_by_threshold = {}
    for threshold in OUTLIER_THRESHOLDS:
        df = outcomes[threshold]
        sub = df[df.bettable_side == side]
        decided = sub[sub.outcome != "push"]
        print(f"\nX = {threshold * 100:.0f}pp (n={len(sub)}, n_decided={len(decided)}):")
        if decided.empty:
            print("  no decided bets")
            profits_by_threshold[threshold] = np.array([])
            continue

        rows = []
        for season, s in decided.groupby("season"):
            rows.append({"season": season, "n": len(s), "roi_pct": s.profit.mean() * 100})
        table = pd.DataFrame(rows).set_index("season")
        print(table.to_string())
        all_positive = bool((table["roi_pct"] > 0).all())
        print(f"  all seasons positive: {all_positive}")

        profits_by_threshold[threshold] = decided["profit"].to_numpy()

    print(f"\n--- {side}-side outliers: by market ---")
    for threshold in OUTLIER_THRESHOLDS:
        df = outcomes[threshold]
        decided = df[(df.bettable_side == side) & (df.outcome != "push")]
        print(f"\nX = {threshold * 100:.0f}pp:")
        if decided.empty:
            print("  no decided bets")
            continue
        rows = []
        for market, s in decided.groupby("market"):
            rows.append({"market": market, "n": len(s), "roi_pct": s.profit.mean() * 100})
        print(pd.DataFrame(rows).set_index("market").sort_values("roi_pct", ascending=False).to_string())

    return profits_by_threshold


def report_bootstrap(profits_by_threshold: dict, side: str) -> None:
    print(f"\n--- {side}-side outliers: bootstrap of pooled ROI (10,000 resamples, seed=0) ---")
    for threshold in OUTLIER_THRESHOLDS:
        profits = profits_by_threshold[threshold]
        if len(profits) == 0:
            print(f"  X={threshold * 100:.0f}pp: no decided bets")
            continue
        means = bootstrap_roi(profits) * 100
        observed = profits.mean() * 100
        p5, p95 = np.percentile(means, [5, 95])
        verdict = ("5th pctile is BELOW ZERO — not statistically distinguishable from a "
                   "losing/breakeven strategy" if p5 < 0 else "5th pctile is ABOVE ZERO")
        print(f"  X={threshold * 100:.0f}pp: n={len(profits)}  observed ROI={observed:+.2f}%  "
              f"5th pctile={p5:+.2f}%  95th pctile={p95:+.2f}%  -> {verdict}")


def report_under_side_baseline(ou: pd.DataFrame, actual_stats: dict) -> None:
    """Mean raw (vig-inclusive) implied probability and mean hit rate for
    EVERY Under quote in the non-scratch dataset (all lines, all books —
    not just outliers), to check whether underpricing is a market-wide
    Under-side property or confined to the outlier subset."""
    print("\n" + "=" * 70)
    print("Under (and Over, for comparison) side, ALL quotes — not just outliers")
    print("=" * 70)

    for side in ("Under", "Over"):
        book_side = ou[ou.side == side]
        parts = []
        for market, sub in book_side.groupby("market"):
            merged = sub.merge(actual_stats[market], on=["game_id", "player_id"], how="left")
            merged["actual"] = merged["actual"].fillna(0.0)
            parts.append(merged)
        df = pd.concat(parts, ignore_index=True)

        if side == "Under":
            outcome = np.where(df.actual == df.line, "push", np.where(df.actual < df.line, "win", "loss"))
        else:
            outcome = np.where(df.actual == df.line, "push", np.where(df.actual > df.line, "win", "loss"))
        df["outcome"] = outcome
        decided = df[df.outcome != "push"]

        print(f"\n{side}: n={len(df)}  n_push={len(df) - len(decided)}  "
              f"mean_raw_implied_p={decided.p_raw.mean():.3f}  hit_rate={(decided.outcome == 'win').mean():.3f}")

        rows = []
        for market, s in decided.groupby("market"):
            rows.append({
                "market": market, "n": len(s), "mean_raw_implied_p": s.p_raw.mean(),
                "hit_rate": (s.outcome == "win").mean(),
            })
        print(pd.DataFrame(rows).set_index("market").to_string())


def run_under_over_deep_dive(ou_results: dict) -> None:
    outcomes = ou_results["outcomes"]

    print("\n" + "=" * 70)
    print("DEEP DIVE — Under-side outliers: season/market robustness + bootstrap")
    print("=" * 70)
    under_profits = report_side_robustness(outcomes, "Under")
    report_bootstrap(under_profits, "Under")

    print("\n" + "=" * 70)
    print("DEEP DIVE — Over-side outliers (mirror check): season/market robustness + bootstrap")
    print("=" * 70)
    over_profits = report_side_robustness(outcomes, "Over")
    report_bootstrap(over_profits, "Over")

    report_under_side_baseline(ou_results["ou"], ou_results["actual_stats"])


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

    ou_results = run_ou_markets()
    run_under_over_deep_dive(ou_results)


if __name__ == "__main__":
    main()
