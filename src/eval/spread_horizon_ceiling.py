"""How much does spread accuracy — and the perfect-foresight ceiling on
CLV — change as you look further before kickoff? Reuses
validate_featured.py's game_id consolidation and consensus/close/
fixed-horizon-open spread builders directly; this module only adds the
per-horizon accuracy and ceiling analysis on top.

ROI throughout assumes flat -110 pricing (featured.parquet does carry
real per-book spread prices, but the request specifically calls for
-110 — the standard assumption for spread bets absent a quoted price).

Sign convention: consensus/close/open spreads are home-perspective, book
convention (home favourite = negative), matching validate_featured.py.
home_cover_margin = margin_home + spread; positive means home covered.

Pure offline analysis — no network calls, only cached parquet files.
"""

import numpy as np
import pandas as pd

from src.eval.book_disagreement import bootstrap_roi
from src.eval.validate_featured import (
    build_event_schedule_map, compute_close_spreads,
    compute_consensus_home_spreads, compute_fixed_horizon_open, load_data,
)

HORIZONS = [14, 10, 7, 5, 3]
VIG_PRICE = 1.0 + 100 / 110  # -110 american, decimal


def build_base_data() -> tuple:
    featured, schedules = load_data()
    event_map = build_event_schedule_map(featured, schedules)
    games = event_map.drop_duplicates(subset="game_id")
    consensus = compute_consensus_home_spreads(featured, event_map)
    close = compute_close_spreads(consensus, games)

    margins = schedules[["game_id", "home_score", "away_score"]].dropna().copy()
    margins["margin_home"] = margins["home_score"] - margins["away_score"]

    return games, consensus, close, margins[["game_id", "margin_home"]]


def analyze_horizon(horizon: int, games: pd.DataFrame, consensus: pd.DataFrame,
                     close: pd.DataFrame, margins: pd.DataFrame) -> pd.DataFrame:
    """game_id, season, open_spread, close_spread, margin_home — games with
    a snapshot at/before kickoff-{horizon}d, a pre-kickoff close, and a
    final score."""
    open_df = compute_fixed_horizon_open(consensus, games, horizon)
    merged = open_df.merge(close[["game_id", "close_spread"]], on="game_id", how="inner")
    merged = merged.merge(margins, on="game_id", how="inner")
    return merged


def report_accuracy(results: dict) -> None:
    print("\n" + "=" * 70)
    print("PART 2 — Accuracy: MAE of horizon spread vs actual margin, and vs close")
    print("=" * 70)
    rows = []
    for h in HORIZONS:
        merged = results[h]
        mae_open_vs_actual = (merged.open_spread + merged.margin_home).abs().mean()
        mae_close_vs_actual = (merged.close_spread + merged.margin_home).abs().mean()
        mae_open_vs_close = (merged.close_spread - merged.open_spread).abs().mean()
        rows.append({
            "horizon_days": h, "n": len(merged),
            "mae_horizon_vs_actual": mae_open_vs_actual,
            "mae_close_vs_actual": mae_close_vs_actual,
            "mae_horizon_vs_close": mae_open_vs_close,
        })
    print(pd.DataFrame(rows).set_index("horizon_days").round(4).to_string())


def report_movement(results: dict) -> None:
    print("\n" + "=" * 70)
    print("PART 3 — Movement: SD of (close_spread - horizon_spread), fraction moving 1+ point")
    print("=" * 70)
    rows = []
    for h in HORIZONS:
        merged = results[h]
        diff = merged.close_spread - merged.open_spread
        rows.append({
            "horizon_days": h, "n": len(diff), "mean_diff": diff.mean(), "sd": diff.std(),
            "frac_moved_1plus": (diff.abs() >= 1).mean(),
        })
    print(pd.DataFrame(rows).set_index("horizon_days").round(4).to_string())


def report_ceiling(results: dict) -> None:
    print("\n" + "=" * 70)
    print("PART 4 — Perfect-foresight ceiling: back the side the CLOSE moved toward, "
          "at the horizon's number, ROI at -110")
    print("=" * 70)

    for h in HORIZONS:
        merged = results[h]
        n_tied = int((merged.close_spread == merged.open_spread).sum())
        df = merged[merged.close_spread != merged.open_spread].copy()

        # spread became MORE negative (home favoured more) -> market moved toward home;
        # back home at the horizon's (more generous) number. Otherwise back away.
        df["bettable_side"] = np.where(df.close_spread < df.open_spread, "home", "away")
        df["home_cover_margin"] = df.margin_home + df.open_spread
        df["outcome"] = np.select(
            [df.home_cover_margin == 0,
             (df.bettable_side == "home") & (df.home_cover_margin > 0),
             (df.bettable_side == "away") & (df.home_cover_margin < 0)],
            ["push", "win", "win"], default="loss")
        df["profit"] = np.where(df.outcome == "push", np.nan,
                                 np.where(df.outcome == "win", VIG_PRICE - 1.0, -1.0))
        decided = df[df.outcome != "push"]

        print(f"\n--- horizon={h}d ---")
        print(f"  n={len(merged)}  n_tied_excluded={n_tied}  n_decided={len(decided)}  "
              f"n_push={len(df) - len(decided)}")
        if decided.empty:
            print("  no decided bets")
            continue

        profits = decided["profit"].to_numpy()
        hit_rate = (decided.outcome == "win").mean()
        roi = profits.mean() * 100
        p5 = np.percentile(bootstrap_roi(profits), 5) * 100 if len(profits) > 1 else float("nan")
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
    games, consensus, close, margins = build_base_data()
    n_total = games.game_id.nunique()
    print(f"Total games: {n_total}\n")

    results = {h: analyze_horizon(h, games, consensus, close, margins) for h in HORIZONS}

    print("=" * 70)
    print("PART 1 — Coverage: games with a snapshot at/before each horizon (before matching to a close/score)")
    print("=" * 70)
    for h in HORIZONS:
        raw_open = compute_fixed_horizon_open(consensus, games, h)
        n_covered = raw_open.game_id.nunique()
        n_matched = results[h].game_id.nunique()
        print(f"  {h:2d}d: {n_covered}/{n_total} games have a horizon snapshot "
              f"({n_covered / n_total * 100:.1f}%)  ->  {n_matched} also have a close + final score")

    report_accuracy(results)
    report_movement(results)
    report_ceiling(results)


if __name__ == "__main__":
    main()
