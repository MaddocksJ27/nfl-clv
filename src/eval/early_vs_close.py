"""Compare the kickoff-72h ("early") snapshot to the kickoff-10min
("close") snapshot for player_receptions and player_rush_attempts: how
much does the line move, how much does the price move when the line
doesn't, which snapshot is actually more accurate, and what's the
theoretical ceiling on CLV if you always knew which way the line would
move?

Joined on (market, game_id, player_id, book) — each book's own PRIMARY
line at each snapshot (see book_disagreement.py's select_primary_lines;
same alt-line rationale applies here). Scratches excluded via the
existing snap_counts definition, applied to both snapshots before the
join.

Pure offline analysis — no network calls, only cached parquet files.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.eval.book_disagreement import (
    bootstrap_roi, compute_actual_ou_stats, compute_is_scratch,
    load_ou_props, select_primary_lines,
)
from src.eval.props_sanity import SKILL_POSITIONS

REPO_ROOT = Path(__file__).resolve().parents[2]
EARLY_PROPS_PATH = REPO_ROOT / "data" / "interim" / "props_early.parquet"
EARLY_EVENT_INDEX_PATH = REPO_ROOT / "data" / "raw" / "props_early" / "_event_index.parquet"

MARKETS = ("player_receptions", "player_rush_attempts")


def load_early_props() -> pd.DataFrame:
    props = pd.read_parquet(EARLY_PROPS_PATH)
    early = props[
        props.market.isin(MARKETS)
        & props.side.isin(["Over", "Under"])
        & props.position.isin(SKILL_POSITIONS)
        & props.player_id.notna()
    ].copy()

    idx = pd.read_parquet(EARLY_EVENT_INDEX_PATH)[
        ["event_id", "game_id", "season", "week", "home_team", "away_team"]]
    early = early.merge(idx, on="event_id", how="inner")
    early["p_raw"] = 1.0 / early["price"]

    early = early[~compute_is_scratch(early)].reset_index(drop=True)
    return early


def load_close_props() -> pd.DataFrame:
    ou = load_ou_props()  # already scratch-excluded, all 5 O/U markets
    return ou[ou.market.isin(MARKETS)].reset_index(drop=True)


def _book_wide(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (market, game_id, player_id, line, book, season): each
    book's PRIMARY line with price_Over/Under, p_raw_Over/Under, and
    p_over_devig (proportional de-vig)."""
    primary = select_primary_lines(df)
    wide = primary.pivot_table(
        index=["market", "game_id", "player_id", "line", "book", "season"],
        columns="side", values=["price", "p_raw"], aggfunc="first",
    )
    wide.columns = [f"{val}_{side}" for val, side in wide.columns]
    wide = wide.reset_index()
    wide["p_over_devig"] = wide["p_raw_Over"] / (wide["p_raw_Over"] + wide["p_raw_Under"])
    return wide


def build_datasets() -> tuple:
    early = _book_wide(load_early_props())
    close = _book_wide(load_close_props())

    merged = early.merge(close, on=["market", "game_id", "player_id", "book"],
                          how="inner", suffixes=("_early", "_close"))
    merged["season"] = merged["season_early"]
    merged = merged.drop(columns=["season_early", "season_close"])

    actual_stats = compute_actual_ou_stats()
    actual_frames = []
    for market in MARKETS:
        a = actual_stats[market].copy()
        a["market"] = market
        actual_frames.append(a)
    actual_all = pd.concat(actual_frames, ignore_index=True)

    merged = merged.merge(actual_all, on=["market", "game_id", "player_id"], how="left")
    merged["actual"] = merged["actual"].fillna(0.0)

    return early, close, merged


def report_coverage(early: pd.DataFrame, close: pd.DataFrame, merged: pd.DataFrame) -> None:
    print("=" * 70)
    print("PART 1 — Coverage: player-game pairs (and book-rows) present at BOTH timestamps")
    print("=" * 70)
    for market in MARKETS:
        e = early[early.market == market]
        c = close[close.market == market]
        m = merged[merged.market == market]
        e_pairs = e.drop_duplicates(["game_id", "player_id"]).shape[0]
        c_pairs = c.drop_duplicates(["game_id", "player_id"]).shape[0]
        m_pairs = m.drop_duplicates(["game_id", "player_id"]).shape[0]

        print(f"\n--- {market} ---")
        print(f"  player-game pairs: early={e_pairs}  close={c_pairs}  matched={m_pairs}  "
              f"(matched/close={m_pairs / c_pairs * 100:.1f}%)")
        print(f"  book-rows:         early={len(e)}  close={len(c)}  matched={len(m)}  "
              f"(matched/close={len(m) / len(c) * 100:.1f}%)")

        print("\n  By season (player-game pairs):")
        rows = []
        for season in sorted(c.season.unique()):
            rows.append({
                "season": season,
                "early": e[e.season == season].drop_duplicates(["game_id", "player_id"]).shape[0],
                "close": c[c.season == season].drop_duplicates(["game_id", "player_id"]).shape[0],
                "matched": m[m.season == season].drop_duplicates(["game_id", "player_id"]).shape[0],
            })
        print(pd.DataFrame(rows).set_index("season").to_string())

        print("\n  By book (book-rows):")
        rows = []
        for book in sorted(c.book.unique()):
            n_early, n_close, n_matched = len(e[e.book == book]), len(c[c.book == book]), len(m[m.book == book])
            rows.append({"book": book, "early": n_early, "close": n_close, "matched": n_matched,
                         "match_rate": n_matched / n_close if n_close else float("nan")})
        print(pd.DataFrame(rows).set_index("book").sort_values("match_rate").round(3).to_string())


def report_line_movement(merged: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PART 2 — Line movement: close_line - early_line, per book-row")
    print("=" * 70)
    merged = merged.copy()
    merged["line_diff"] = merged["line_close"] - merged["line_early"]
    for market in MARKETS:
        sub = merged.loc[merged.market == market, "line_diff"]
        print(f"\n--- {market} (n={len(sub)}) ---")
        print(f"  mean={sub.mean():+.3f}  sd={sub.std():.3f}  "
              f"frac_unchanged={(sub == 0).mean() * 100:.1f}%  frac_moved_1plus={(sub.abs() >= 1).mean() * 100:.1f}%")
        print(sub.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_string())


def report_price_movement(merged: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PART 3 — De-vigged Over probability change (72h -> close), lines UNCHANGED only")
    print("=" * 70)
    unchanged = merged[merged.line_close == merged.line_early].copy()
    unchanged["p_over_devig_diff"] = unchanged["p_over_devig_close"] - unchanged["p_over_devig_early"]
    for market in MARKETS:
        sub = unchanged.loc[unchanged.market == market, "p_over_devig_diff"]
        print(f"\n--- {market} (n={len(sub)}) ---")
        print(f"  mean={sub.mean():+.4f}  sd={sub.std():.4f}")
        print(sub.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_string())


def report_mae_comparison(merged: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PART 4 — Which line is better? MAE vs actual outcome, paired test on the difference")
    print("=" * 70)
    merged = merged.copy()
    merged["ae_early"] = (merged["line_early"] - merged["actual"]).abs()
    merged["ae_close"] = (merged["line_close"] - merged["actual"]).abs()

    for market in MARKETS:
        sub = merged[merged.market == market]
        mae_early, mae_close = sub["ae_early"].mean(), sub["ae_close"].mean()
        t_stat, p_value = scipy_stats.ttest_rel(sub["ae_early"], sub["ae_close"])
        w_stat, w_p = scipy_stats.wilcoxon(sub["ae_early"], sub["ae_close"])

        print(f"\n--- {market} (n={len(sub)}) ---")
        print(f"  MAE early={mae_early:.4f}  MAE close={mae_close:.4f}  "
              f"diff(early-close)={mae_early - mae_close:+.4f}  "
              f"({'close more accurate' if mae_close < mae_early else 'early more accurate'})")
        print(f"  paired t-test:          t={t_stat:.3f}  p={p_value:.4g}")
        print(f"  Wilcoxon signed-rank:   W={w_stat:.1f}  p={w_p:.4g}")


def report_upper_bound_clv(merged: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PART 5 — Upper bound on CLV: back the side the CLOSING line moved toward, "
          "at the 72h price (deliberate hindsight)")
    print("=" * 70)
    df = merged.copy()
    n_tied = int((df.line_close == df.line_early).sum())
    df = df[df.line_close != df.line_early].copy()
    print(f"\n({n_tied} book-rows with an unchanged line excluded — no movement signal to exploit)")

    df["bettable_side"] = np.where(df.line_close > df.line_early, "Over", "Under")
    df["bet_price"] = np.where(df.bettable_side == "Over", df["price_Over_early"], df["price_Under_early"])
    df["outcome"] = np.select(
        [df.actual == df.line_early,
         (df.bettable_side == "Over") & (df.actual > df.line_early),
         (df.bettable_side == "Under") & (df.actual < df.line_early)],
        ["push", "win", "win"], default="loss")
    df["profit"] = np.where(df.outcome == "push", np.nan,
                             np.where(df.outcome == "win", df["bet_price"] - 1.0, -1.0))

    for market in MARKETS:
        sub = df[df.market == market]
        decided = sub[sub.outcome != "push"]
        profits = decided["profit"].to_numpy()
        hit_rate = (decided.outcome == "win").mean()
        roi = profits.mean() * 100
        p5 = np.percentile(bootstrap_roi(profits), 5) * 100 if len(profits) > 1 else float("nan")

        print(f"\n--- {market}: n={len(sub)}  n_decided={len(decided)} ---")
        print(f"  POOLED: hit_rate={hit_rate:.3f}  ROI={roi:+.2f}%  boot_5th_pctile={p5:+.2f}%")

        rows = []
        for season, s in decided.groupby("season"):
            s_profits = s["profit"].to_numpy()
            s_hit_rate = (s.outcome == "win").mean()
            s_roi = s_profits.mean() * 100
            s_p5 = np.percentile(bootstrap_roi(s_profits), 5) * 100 if len(s_profits) > 1 else float("nan")
            rows.append({"season": season, "n": len(s), "hit_rate": s_hit_rate,
                         "roi_pct": s_roi, "boot_5th_pctile": s_p5})
        print(pd.DataFrame(rows).set_index("season").round(4).to_string())


def main() -> None:
    early, close, merged = build_datasets()
    print(f"Early (72h) book-rows: {len(early)}   Close (10min) book-rows: {len(close)}   "
          f"Matched: {len(merged)}\n")

    report_coverage(early, close, merged)
    report_line_movement(merged)
    report_price_movement(merged)
    report_mae_comparison(merged)
    report_upper_bound_clv(merged)


if __name__ == "__main__":
    main()
