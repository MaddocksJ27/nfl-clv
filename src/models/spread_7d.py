"""Predict the DIRECTION of the subsequent spread move from a 7-day
horizon: target = sign(close_spread - consensus_spread_at_kickoff_minus_7d).

Reuses drift.py's book-disagreement and model_1 OOF-prediction machinery
(both horizon-agnostic already) and spread_baseline.py's model_1 feature
set wholesale — this module only adds: the 7-day horizon itself, a
consensus TOTAL feature (new — drift.py dropped totals entirely when it
cut Group A down to 4 features; this model's feature list explicitly
calls for one), a primetime flag (new), and the classification +
betting-ROI layer on top.

CRITICAL no-lookahead rule: every feature must be knowable at kickoff-7d.

Features, all knowable at kickoff-7d:
  - model_1's predicted CLOSE spread minus the kickoff-7d consensus
  - consensus spread and consensus total at kickoff-7d
  - book disagreement (SD of spread across books) at kickoff-7d
  - the 12 model_1 features: 4 no-split EPA ratings (home/away off/def),
    home/away rest days, div_game, week, home/away QB rating, home/away
    QB change (numeric); home/away QB injury status (categorical)
  - primetime flag (kickoff hour >= 19:00 local, any weekday — covers
    MNF/TNF/SNF/Saturday-night/int'l-night games in one simple rule)

model_1 is refit walk-forward with the SAME season folds used here, so
its predictions for a test season never came from a model that saw that
season or later.

Games where the spread doesn't move at all by close (target == 0) are
dropped — sign is undefined, no direction to classify.

ROI assumes flat -110 pricing (see spread_horizon_ceiling.py, whose
push/cover-margin convention this module reuses directly).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint

from src.eval.book_disagreement import bootstrap_roi
from src.eval.spread_horizon_ceiling import VIG_PRICE
from src.eval.validate_featured import (
    build_event_schedule_map, compute_close_spreads, compute_consensus_home_spreads,
    compute_fixed_horizon_open, load_data,
)
from src.models.drift import compute_book_disagreement, compute_model1_oof_predictions
from src.models.spread_baseline import ALL_CATEGORICAL_FEATURES as MODEL1_ALL_CATEGORICAL_FEATURES
from src.models.spread_baseline import NO_SPLIT_NUMERIC_FEATURES as MODEL1_NUMERIC_FEATURES
from src.models.spread_baseline import build_dataset as build_model1_dataset

HORIZON_DAYS = 7
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
TOP_QUARTILE = 0.75  # confidence percentile cutoff

Z_ALPHA_ONE_SIDED = norm.ppf(0.95)  # 1.6449 -- matches the 5th-percentile bootstrap threshold used throughout
POWER_TEST_HIT_RATES = (0.53, 0.54, 0.55)
POWER_N_SIMS = 10_000
TARGET_POWER = 0.80

NUMERIC_FEATURES = (
    ["model1_minus_consensus", "consensus_spread_h7", "consensus_total_h7", "book_disagreement_h7"]
    + MODEL1_NUMERIC_FEATURES + ["primetime"]
)
CATEGORICAL_FEATURES = ["home_qb_injury", "away_qb_injury"]


def compute_consensus_totals(featured: pd.DataFrame, event_map: pd.DataFrame) -> pd.DataFrame:
    """game_id, snapshot_time, consensus_total — mean total line across
    books, per snapshot (Over/Under share one number, so restrict to the
    Over rows to avoid double-counting each book)."""
    totals = featured[featured.market == "totals"]
    game_lookup = event_map.set_index("event_id")["game_id"]

    tagged = totals[totals.event_id.isin(game_lookup.index)].copy()
    tagged["game_id"] = tagged.event_id.map(game_lookup)
    over_rows = tagged[tagged.team == "Over"]

    consensus = over_rows.groupby(["game_id", "snapshot_time"]).line.mean().reset_index()
    consensus.columns = ["game_id", "snapshot_time", "consensus_total"]
    consensus["snapshot_ts"] = pd.to_datetime(consensus.snapshot_time, utc=True)
    return consensus


def compute_fixed_horizon_total(consensus_totals: pd.DataFrame, games: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """game_id, total_h{horizon} — consensus total at the latest snapshot
    at/before kickoff-{horizon_days}d. Mirrors validate_featured.py's
    compute_fixed_horizon_open, for totals instead of spreads."""
    merged = consensus_totals.merge(games[["game_id", "commence_ts"]], on="game_id")
    cutoff = merged.commence_ts - pd.Timedelta(days=horizon_days)
    eligible = merged[merged.snapshot_ts <= cutoff]
    latest = eligible.sort_values("snapshot_ts").groupby("game_id").tail(1).copy()
    return latest.rename(columns={"consensus_total": f"total_h{horizon_days}"})[["game_id", f"total_h{horizon_days}"]]


def build_dataset() -> pd.DataFrame:
    """One row per game with a valid 7-day-horizon open, a pre-kickoff
    close, a final score, and every feature. target = continuous drift
    (close - horizon spread); y = 1 if target > 0 else 0. Rows with
    target == 0 (no movement — undefined sign) are dropped."""
    featured, schedules = load_data()
    event_map = build_event_schedule_map(featured, schedules)
    games = event_map.drop_duplicates(subset="game_id")

    consensus = compute_consensus_home_spreads(featured, event_map)
    close = compute_close_spreads(consensus, games)
    horizon7 = compute_fixed_horizon_open(consensus, games, HORIZON_DAYS)

    disagreement = compute_book_disagreement(featured, event_map, horizon7).rename(
        columns={"book_disagreement_h3": "book_disagreement_h7"})

    consensus_totals = compute_consensus_totals(featured, event_map)
    total_h7 = compute_fixed_horizon_total(consensus_totals, games, HORIZON_DAYS)

    df = close.merge(horizon7, on=["game_id", "season"]).rename(columns={"open_spread": "consensus_spread_h7"})
    df["target"] = df["close_spread"] - df["consensus_spread_h7"]
    df = df.merge(disagreement, on="game_id", how="left")
    df = df.merge(total_h7, on="game_id", how="left").rename(columns={"total_h7": "consensus_total_h7"})

    model1_df = build_model1_dataset()
    context_cols = ["game_id", "home_team", "away_team"] + MODEL1_NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df = df.merge(model1_df[context_cols], on="game_id", how="left")

    oof = compute_model1_oof_predictions(model1_df)
    df = df.merge(oof, on="game_id", how="left")
    df["model1_minus_consensus"] = df["model1_pred_spread"] - df["consensus_spread_h7"]

    schedules["primetime"] = (schedules["gametime"].str.slice(0, 2).astype(int) >= 19).astype(int)
    df = df.merge(schedules[["game_id", "primetime"]], on="game_id", how="left")

    margins = schedules[["game_id", "home_score", "away_score"]].dropna().copy()
    margins["margin_home"] = margins["home_score"] - margins["away_score"]
    df = df.merge(margins[["game_id", "margin_home"]], on="game_id", how="left")

    required = ["target", "margin_home"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    n_before = len(df)
    df = df.dropna(subset=required)
    n_dropped_missing = n_before - len(df)

    n_before_ties = len(df)
    df = df[df["target"] != 0].copy()
    n_ties = n_before_ties - len(df)

    if n_dropped_missing:
        print(f"Dropped {n_dropped_missing}/{n_before} games with missing target/feature/score values")
    print(f"Dropped {n_ties}/{n_before_ties} games with target==0 (no movement — sign undefined)")

    df = df.reset_index(drop=True)
    df["season"] = df["season"].astype(int)
    df["y"] = (df["target"] > 0).astype(int)
    df["home_cover_margin"] = df["margin_home"] + df["consensus_spread_h7"]

    for col in CATEGORICAL_FEATURES:
        df[col] = pd.Categorical(df[col], categories=sorted(df[col].astype(str).unique()))

    return df


def walk_forward_logistic_ridge(df: pd.DataFrame) -> pd.DataFrame:
    """Ridge-penalized (L2) logistic regression, walk-forward by season.
    C (=1/alpha) is selected by LogisticRegressionCV's internal
    cross-validation, run only on the training fold passed to .fit() —
    the test fold is never touched until after C is fixed. Returns one
    row per test-fold game: game_id, season, y_true, y_pred, proba,
    confidence."""
    dummies = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=False)
    X_full = pd.concat([df[NUMERIC_FEATURES].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    feature_cols = X_full.columns.tolist()

    seasons = sorted(df.season.unique())
    rows = []

    for season in seasons[1:]:
        train_mask = (df.season < season).to_numpy()
        test_mask = (df.season == season).to_numpy()
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        y_train = df.loc[train_mask, "y"].to_numpy()
        if len(set(y_train)) < 2:
            continue

        X_train = X_full.loc[train_mask, feature_cols].to_numpy(dtype=float)
        X_test = X_full.loc[test_mask, feature_cols].to_numpy(dtype=float)

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegressionCV(
            Cs=1.0 / RIDGE_ALPHAS, cv=5, penalty="l2", solver="lbfgs",
            max_iter=5000, scoring="neg_log_loss",
        )
        clf.fit(X_train_s, y_train)
        proba = clf.predict_proba(X_test_s)[:, 1]
        pred = (proba >= 0.5).astype(int)

        test_df = df.loc[test_mask]
        rows.append(pd.DataFrame({
            "game_id": test_df["game_id"].to_numpy(), "season": season,
            "y_true": test_df["y"].to_numpy(), "y_pred": pred, "proba": proba,
            "confidence": np.maximum(proba, 1 - proba),
        }))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["game_id", "season", "y_true", "y_pred", "proba", "confidence"])


def report_accuracy(oof: pd.DataFrame) -> None:
    print("=" * 70)
    print("DIRECTIONAL ACCURACY vs 50% baseline")
    print("=" * 70)
    rows = []
    for season, s in oof.groupby("season"):
        acc = (s.y_true == s.y_pred).mean()
        rows.append({"season": season, "n": len(s), "accuracy": acc})
    print(pd.DataFrame(rows).set_index("season").round(4).to_string())
    overall = (oof.y_true == oof.y_pred).mean()
    print(f"\nPOOLED: n={len(oof)}  accuracy={overall:.4f}  "
          f"vs 50% baseline: {'BEATS' if overall > 0.5 else 'does NOT beat'}")


def _bet_outcomes(oof: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Merges predictions with home_cover_margin, determines the bettable
    side per the MODEL's predicted direction (y_pred==1 -> spread will
    move toward away winning value -> back away at the horizon's number;
    y_pred==0 -> back home), computes win/loss/push and profit at -110."""
    merged = oof.merge(df[["game_id", "home_cover_margin"]], on="game_id", how="left")
    merged["bettable_side"] = np.where(merged.y_pred == 1, "away", "home")
    merged["outcome"] = np.select(
        [merged.home_cover_margin == 0,
         (merged.bettable_side == "home") & (merged.home_cover_margin > 0),
         (merged.bettable_side == "away") & (merged.home_cover_margin < 0)],
        ["push", "win", "win"], default="loss")
    merged["profit"] = np.where(merged.outcome == "push", np.nan,
                                 np.where(merged.outcome == "win", VIG_PRICE - 1.0, -1.0))
    return merged


def _report_roi(bets: pd.DataFrame, label: str) -> None:
    decided = bets[bets.outcome != "push"]
    n_push = len(bets) - len(decided)
    print(f"\n{label}: n={len(bets)}  n_push={n_push}  n_decided={len(decided)}")
    if decided.empty:
        print("  no decided bets")
        return

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


def report_roi(oof: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("ROI at -110, betting every predicted direction")
    print("=" * 70)
    bets = _bet_outcomes(oof, df)
    _report_roi(bets, "ALL predicted bets")

    print("\n" + "=" * 70)
    print(f"ROI at -110, restricted to the top {(1 - TOP_QUARTILE) * 100:.0f}% by predicted-direction confidence")
    print("=" * 70)
    threshold = oof["confidence"].quantile(TOP_QUARTILE)
    top = bets[bets["confidence"] >= threshold]
    print(f"(confidence threshold: {threshold:.4f})")
    _report_roi(top, "TOP-QUARTILE-CONFIDENCE bets")

    return bets


def _profit_mean_sd(hit_rate: float) -> tuple:
    """Population mean and SD of per-bet profit at -110, for a bet with
    the given TRUE hit rate — exact, from the two-point {win, loss}
    distribution (win=+100/110, loss=-1)."""
    win_profit = VIG_PRICE - 1.0
    loss_profit = -1.0
    mean = hit_rate * win_profit + (1 - hit_rate) * loss_profit
    sd = np.sqrt(hit_rate * (1 - hit_rate)) * (win_profit - loss_profit)
    return mean, sd


def simulate_detection_power(true_hit_rate: float, n: int, n_sims: int = POWER_N_SIMS, seed: int = 0) -> float:
    """Fraction of n_sims simulated n-bet samples at the given TRUE hit
    rate whose bootstrap 5th percentile would land above zero.

    Uses a normal approximation to the percentile bootstrap
    (bootstrap 5th pctile ~= sample_mean - z_0.95 * sample_SE) rather than
    a literal nested bootstrap (10,000 outer sims x 10,000 inner
    resamples would be ~1e8 resamples per hit rate — intractable). This
    is justified by the CLT for a bounded two-point profit distribution
    at n in the hundreds, and was checked directly: on the module's own
    730-bet sample, this formula gives -6.92%, versus book_disagreement's
    actual bootstrap_roi() giving -6.90% on the identical data — a
    near-exact match."""
    rng = np.random.default_rng(seed)
    win_profit = VIG_PRICE - 1.0
    loss_profit = -1.0

    wins = rng.binomial(n, true_hit_rate, size=n_sims)
    sample_hit_rate = wins / n
    sample_mean = sample_hit_rate * win_profit + (1 - sample_hit_rate) * loss_profit
    sample_var = (wins * (win_profit - sample_mean) ** 2
                  + (n - wins) * (loss_profit - sample_mean) ** 2) / (n - 1)
    se = np.sqrt(sample_var / n)
    approx_p5 = sample_mean - Z_ALPHA_ONE_SIDED * se

    return float((approx_p5 > 0).mean())


def required_sample_size(true_hit_rate: float, target_power: float = TARGET_POWER) -> float:
    """Closed-form n for `target_power` one-sided power (alpha matching the
    z=1.6449 / 5th-percentile threshold used throughout this file) to
    distinguish a true hit rate from break-even at -110: the standard
    n = ((z_alpha + z_beta) * sigma / mu)^2 formula for a one-sample mean
    test against a null of zero."""
    mean, sd = _profit_mean_sd(true_hit_rate)
    z_beta = norm.ppf(target_power)
    return ((Z_ALPHA_ONE_SIDED + z_beta) * sd / mean) ** 2


def report_power_analysis(bets: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("POWER ANALYSIS — can this sample size detect a real edge if one exists?")
    print("=" * 70)

    n_bets = len(bets)
    print(f"\n1. Simulated detection power at n={n_bets} bets, {POWER_N_SIMS:,} simulated samples per true hit rate:")
    for p in POWER_TEST_HIT_RATES:
        power = simulate_detection_power(p, n_bets)
        mean, _ = _profit_mean_sd(p)
        print(f"  true hit rate={p:.2f} (true ROI={mean * 100:+.2f}%): "
              f"P(bootstrap 5th pctile > 0) = {power:.3f}")

    print(f"\n2. Sample size required for {TARGET_POWER * 100:.0f}% power to distinguish from break-even at -110:")
    for p in POWER_TEST_HIT_RATES:
        n_needed = required_sample_size(p)
        print(f"  true hit rate={p:.2f}: n={n_needed:,.0f}")

    print("\n3. 95% CI on the observed hit rate, expressed as ROI bounds:")
    decided = bets[bets.outcome != "push"]
    n_decided = len(decided)
    n_wins = int((decided.outcome == "win").sum())
    p_hat = n_wins / n_decided
    ci_low, ci_high = proportion_confint(n_wins, n_decided, alpha=0.05, method="wilson")

    roi_point, _ = _profit_mean_sd(p_hat)
    roi_low, _ = _profit_mean_sd(ci_low)
    roi_high, _ = _profit_mean_sd(ci_high)
    print(f"  observed: n={n_decided}  hit_rate={p_hat:.4f}  point ROI={roi_point * 100:+.2f}%")
    print(f"  95% Wilson CI on hit rate: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  95% CI on ROI:            [{roi_low * 100:+.2f}%, {roi_high * 100:+.2f}%]")


def main() -> None:
    df = build_dataset()
    print(f"\nDataset: {len(df)} games, seasons {sorted(df.season.unique())}")
    print(f"Direction balance: away_gained(y=1)={df.y.mean():.3f}  home_gained(y=0)={1 - df.y.mean():.3f}")

    oof = walk_forward_logistic_ridge(df)
    report_accuracy(oof)
    bets = report_roi(oof, df)
    report_power_analysis(bets)


if __name__ == "__main__":
    main()
