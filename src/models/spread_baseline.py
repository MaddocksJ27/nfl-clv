"""BASELINE spread model (see CLAUDE.md): predict the consensus closing
spread (home perspective, book convention — home favourite negative) from
opponent-adjusted team strength. Its real job is producing implied team
points for the anytime-TD layer; a high R^2 here is expected and desired.

Target: closing consensus spread from data/interim/featured.parquet (via
src.eval.validate_featured's event/schedule join and close-spread
computation — not nflverse's own spread_line, which is a different book's
estimate used elsewhere purely as a cross-check).

Features: the six EPA ratings (data/interim/epa_ratings.parquet) for home
and away as-of that week, rest days for each side, div_game, roof, surface,
and week number — all from data/raw/schedules.parquet plus epa_ratings.

Validation: walk-forward by season only (train on all seasons strictly
before S, predict S) — no random splits, since a random split would leak
future seasons' games into training for a past season's prediction.

featured.parquet only covers 2020-2025-02, so usable seasons here are
2020-2024; 2020 itself can't be a test fold (no earlier season to train on).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.eval.validate_featured import (
    build_event_schedule_map, compute_close_spreads, compute_consensus_home_spreads, load_data,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EPA_RATINGS_PATH = REPO_ROOT / "data" / "interim" / "epa_ratings.parquet"

EPA_SPLITS = ("overall", "pass", "run")
EPA_COLS = [f"{split}_{side}_epa" for split in EPA_SPLITS for side in ("off", "def")]

NUMERIC_FEATURES = [f"home_{c}" for c in EPA_COLS] + [f"away_{c}" for c in EPA_COLS] + [
    "home_rest", "away_rest", "div_game", "week",
]
CATEGORICAL_FEATURES = ["roof", "surface"]

LGBM_PARAMS = dict(
    n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1,
)


def build_dataset() -> pd.DataFrame:
    """One row per game with a valid pre-kickoff close: target close_spread
    plus all model features. Restricted to games where the EPA ratings and
    schedule fields are all available."""
    featured, schedules = load_data()
    event_map = build_event_schedule_map(featured, schedules)
    games = event_map.drop_duplicates(subset="game_id")

    consensus = compute_consensus_home_spreads(featured, event_map)
    close = compute_close_spreads(consensus, games)

    df = close.merge(
        games[["game_id", "week", "home_team", "away_team"]], on="game_id",
    ).rename(columns={"close_spread": "target_spread"})

    sched_cols = ["season", "week", "home_team", "away_team", "home_rest", "away_rest", "div_game", "roof", "surface"]
    sched = schedules[sched_cols].copy()
    sched["surface"] = sched["surface"].str.strip()
    df = df.merge(sched, on=["season", "week", "home_team", "away_team"], how="left")

    epa = pd.read_parquet(EPA_RATINGS_PATH)
    home_epa = epa.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in EPA_COLS}})
    away_epa = epa.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in EPA_COLS}})
    df = df.merge(home_epa, on=["season", "week", "home_team"], how="left")
    df = df.merge(away_epa, on=["season", "week", "away_team"], how="left")

    df["roof"] = df["roof"].fillna("unknown")
    df["surface"] = df["surface"].fillna("unknown")

    required = ["target_spread"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    n_before = len(df)
    df = df.dropna(subset=required)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"Dropped {n_dropped}/{n_before} games with missing target/feature values")

    df = df.reset_index(drop=True)
    df["season"] = df["season"].astype(int)

    # Fix categorical levels to the full dataset now, before any fold split —
    # a rare category (e.g. roof='open', 48/1391 rows) can otherwise be
    # entirely absent from a training fold, and patsy/get_dummies would then
    # build a different design matrix at predict time than at fit time.
    for col in CATEGORICAL_FEATURES:
        df[col] = pd.Categorical(df[col], categories=sorted(df[col].unique()))

    return df


def _metrics(y_true, y_pred) -> dict:
    return {
        "n": len(y_true),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _print_metrics_table(per_season: dict, pooled: dict, label: str) -> None:
    print(f"\n{label} — walk-forward by season")
    print(f"  {'season':>8}  {'n':>5}  {'rmse':>7}  {'mae':>7}  {'r2':>8}")
    for season, m in sorted(per_season.items()):
        print(f"  {season:>8}  {m['n']:>5}  {m['rmse']:>7.3f}  {m['mae']:>7.3f}  {m['r2']:>8.3f}")
    print(f"  {'POOLED':>8}  {pooled['n']:>5}  {pooled['rmse']:>7.3f}  {pooled['mae']:>7.3f}  {pooled['r2']:>8.3f}")


def _ols_formula() -> str:
    terms = NUMERIC_FEATURES + [f"C({c})" for c in CATEGORICAL_FEATURES]
    return "target_spread ~ " + " + ".join(terms)


def fit_full_ols(df: pd.DataFrame):
    """OLS on the full dataset, purely for a readable coefficient table —
    not used for the walk-forward metrics (see walk_forward_ols)."""
    model = smf.ols(_ols_formula(), data=df).fit()
    return model


def walk_forward_ols(df: pd.DataFrame):
    formula = _ols_formula()
    seasons = sorted(df.season.unique())
    per_season = {}
    all_true, all_pred = [], []

    for season in seasons[1:]:  # first season has no strictly-earlier training data
        train = df[df.season < season]
        test = df[df.season == season]
        if train.empty or test.empty:
            continue
        model = smf.ols(formula, data=train).fit()
        pred = model.predict(test)
        per_season[season] = _metrics(test.target_spread, pred)
        all_true.extend(test.target_spread.tolist())
        all_pred.extend(pred.tolist())

    pooled = _metrics(pd.Series(all_true), pd.Series(all_pred))
    return per_season, pooled


def walk_forward_lgbm(df: pd.DataFrame):
    dummies = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=False)
    X_full = pd.concat([df[NUMERIC_FEATURES], dummies], axis=1)
    y_full = df["target_spread"]

    seasons = sorted(df.season.unique())
    per_season = {}
    all_true, all_pred = [], []

    for season in seasons[1:]:
        train_mask = df.season < season
        test_mask = df.season == season
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_full[train_mask], y_full[train_mask])
        pred = model.predict(X_full[test_mask])
        per_season[season] = _metrics(y_full[test_mask], pred)
        all_true.extend(y_full[test_mask].tolist())
        all_pred.extend(pred.tolist())

    pooled = _metrics(pd.Series(all_true), pd.Series(all_pred))
    return per_season, pooled


def main() -> None:
    df = build_dataset()
    print(f"Dataset: {len(df)} games, seasons {sorted(df.season.unique())}")

    full_model = fit_full_ols(df)
    print("\n" + "=" * 70)
    print("OLS coefficients (fit on all available games, for interpretation only)")
    print("=" * 70)
    print(full_model.summary())

    ols_per_season, ols_pooled = walk_forward_ols(df)
    _print_metrics_table(ols_per_season, ols_pooled, "OLS")

    lgbm_per_season, lgbm_pooled = walk_forward_lgbm(df)
    _print_metrics_table(lgbm_per_season, lgbm_pooled, "LightGBM")

    print("\n" + "=" * 70)
    print(f"Pooled R^2 — OLS: {ols_pooled['r2']:.3f}   LightGBM: {lgbm_pooled['r2']:.3f}")
    print(f"Pooled RMSE — OLS: {ols_pooled['rmse']:.3f}   LightGBM: {lgbm_pooled['rmse']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
