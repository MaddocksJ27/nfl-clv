"""HEADLINE drift model (see CLAUDE.md): predict open-to-close spread
drift = consensus_close_spread - consensus_spread_at(kickoff-3d), home
perspective (book convention, home favourite negative). Small signal
expected — ~10-20% variance explained is a good result for this target,
unlike BASELINE's high R^2.

Uses the 3-day fixed horizon established in validate_featured.py (1,380
games, drift SD 0.955) as "open" for this model specifically — not the
loose first-ever-observed open used in validate_featured's own drift table.

CRITICAL no-lookahead rule: every feature must be knowable at kickoff-3d.
Nothing from after that point (including the close itself, and anything
from days 0-2 before kickoff) may leak into a feature.

Features (deliberately narrow — an earlier version required a kickoff-10d
snapshot for a 7-day-change/key-number feature pair, which only 711/1,380
games have, and put the full EPA/QB block in Group A, giving ~20 features
against walk-forward training folds as small as ~55 rows and producing
negative R^2 across the board):
  GROUP A — state at the horizon (4 features only):
    model_1's predicted CLOSE spread minus the kickoff-3d consensus (how
    far the market sits from the fundamental estimate), consensus spread
    at kickoff-3d, book disagreement (SD of spread across books) at
    kickoff-3d, week number.
  GROUP B — movement already underway at the horizon:
    spread change over the 2 days before the horizon, whether the line
    crossed a key number (3 or 7) in that same 2-day window (redefined
    from the original 7-day window specifically to drop the kickoff-10d
    dependency and recover sample size).

model_1 (src.models.spread_baseline's recommended no-split OLS) is refit
walk-forward with the SAME season folds used here, so its predictions
never see a season this module is being evaluated on either — model_1
itself is unchanged (still OLS); only THIS module's own Group A/B/combined
regressions use Ridge, with alpha selected by cross-validation on the
training fold only (RidgeCV never sees the test fold's alpha-selection
data, let alone its labels).
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.eval.validate_featured import (
    build_event_schedule_map, compute_close_spreads, compute_consensus_home_spreads,
    compute_fixed_horizon_open, load_data,
)
from src.models.spread_baseline import ALL_CATEGORICAL_FEATURES as MODEL1_CATEGORICAL_FEATURES
from src.models.spread_baseline import NO_SPLIT_NUMERIC_FEATURES as MODEL1_NUMERIC_FEATURES
from src.models.spread_baseline import build_dataset as build_model1_dataset

HORIZON_DAYS = 3
SHORT_BEFORE_DAYS = 2  # for the 2-day-before-horizon change and key-number-crossing window
KEY_NUMBERS = (3, 7)
RIDGE_ALPHAS = np.logspace(-3, 3, 13)

GROUP_A_NUMERIC = ["model1_minus_consensus", "consensus_spread_h3", "book_disagreement_h3", "week"]
GROUP_B_NUMERIC = ["spread_chg_2d", "crossed_key_3", "crossed_key_7"]
ALL_NUMERIC = GROUP_A_NUMERIC + GROUP_B_NUMERIC


def compute_book_disagreement(featured: pd.DataFrame, event_map: pd.DataFrame, horizon_open: pd.DataFrame) -> pd.DataFrame:
    """SD of the home-team spread `line` across books, at each game's own
    horizon snapshot (horizon_open's open_ts) — not the mean (that's the
    consensus itself), the spread of opinion around it."""
    spreads = featured[featured.market == "spreads"]
    game_lookup = event_map.set_index("event_id")["game_id"]
    home_lookup = event_map.set_index("event_id")["home_team_full"]

    tagged = spreads[spreads.event_id.isin(game_lookup.index)].copy()
    tagged["game_id"] = tagged.event_id.map(game_lookup)
    tagged["home_team_full"] = tagged.event_id.map(home_lookup)
    tagged["snapshot_ts"] = pd.to_datetime(tagged.snapshot_time, utc=True)
    home_rows = tagged[tagged.team == tagged.home_team_full]

    target_ts = horizon_open.set_index("game_id")["open_ts"]
    home_rows = home_rows[home_rows.game_id.isin(target_ts.index)].copy()
    home_rows["target_ts"] = home_rows.game_id.map(target_ts)
    at_horizon = home_rows[home_rows.snapshot_ts == home_rows.target_ts]

    disagreement = at_horizon.groupby("game_id").line.std().reset_index()
    disagreement.columns = ["game_id", "book_disagreement_h3"]
    disagreement["book_disagreement_h3"] = disagreement["book_disagreement_h3"].fillna(0.0)
    return disagreement


def compute_key_number_crossings(consensus: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """crossed_key_3 / crossed_key_7: 1 if the consensus home spread's
    min/max within (kickoff-5d, kickoff-3d] straddle +-3 / +-7, else 0 —
    a same-day-granularity proxy for "did the line cross this key number
    in the 2 days before the horizon" (can't see intra-day crossings the
    daily snapshot cadence didn't capture). Deliberately the same 2-day
    window as spread_chg_2d, not the original 7-day window — that needed
    a kickoff-10d snapshot, which only 711/1,380 games have."""
    merged = consensus.merge(games[["game_id", "commence_ts"]], on="game_id")
    upper = merged.commence_ts - pd.Timedelta(days=HORIZON_DAYS)
    lower = upper - pd.Timedelta(days=SHORT_BEFORE_DAYS)
    window = merged[(merged.snapshot_ts > lower) & (merged.snapshot_ts <= upper)]

    grp = window.groupby("game_id").consensus_home_spread.agg(["min", "max"])
    out = pd.DataFrame(index=grp.index)
    for k in KEY_NUMBERS:
        out[f"crossed_key_{k}"] = (
            ((grp["min"] < -k) & (grp["max"] > -k)) | ((grp["min"] < k) & (grp["max"] > k))
        ).astype(int)
    return out.reset_index()


def _formula(numeric_features, categorical_features) -> str:
    terms = numeric_features + [f"C({c})" for c in categorical_features]
    return "target ~ " + " + ".join(terms)


def compute_model1_oof_predictions(model1_df: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold model_1 (spread_baseline's no-split OLS) predictions for
    every game, refit with the SAME walk-forward folds this module uses
    (train strictly-earlier seasons) — model_1's prediction for a game in
    season S never comes from a model that saw season S or later. Games in
    the first season present have no OOF prediction."""
    df = model1_df.rename(columns={"target_spread": "target"})
    formula = _formula(MODEL1_NUMERIC_FEATURES, MODEL1_CATEGORICAL_FEATURES)
    seasons = sorted(df.season.unique())

    preds = []
    for season in seasons[1:]:
        train = df[df.season < season]
        test = df[df.season == season]
        model = smf.ols(formula, data=train).fit()
        pred = model.predict(test)
        preds.append(pd.DataFrame({"game_id": test.game_id.values, "model1_pred_spread": pred.values}))

    return pd.concat(preds, ignore_index=True)


def build_dataset() -> pd.DataFrame:
    """One row per game with a valid 3-day-horizon open: target drift plus
    Group A and Group B features, all knowable at kickoff-3d. No longer
    needs a kickoff-10d snapshot at all (dropped along with spread_chg_7d
    and the 7-day key-number window), so this recovers much closer to the
    full 1,380-game 3-day-horizon population than the first version's 618."""
    featured, schedules = load_data()
    event_map = build_event_schedule_map(featured, schedules)
    games = event_map.drop_duplicates(subset="game_id")

    consensus = compute_consensus_home_spreads(featured, event_map)
    close = compute_close_spreads(consensus, games)
    horizon3 = compute_fixed_horizon_open(consensus, games, HORIZON_DAYS)
    horizon5 = compute_fixed_horizon_open(consensus, games, HORIZON_DAYS + SHORT_BEFORE_DAYS)

    disagreement = compute_book_disagreement(featured, event_map, horizon3)
    crossings = compute_key_number_crossings(consensus, games)

    df = close.merge(horizon3, on=["game_id", "season"]).rename(columns={"open_spread": "consensus_spread_h3"})
    df["target"] = df["close_spread"] - df["consensus_spread_h3"]

    df = df.merge(disagreement, on="game_id", how="left")

    h5 = horizon5.rename(columns={"open_spread": "consensus_spread_h5"})[["game_id", "consensus_spread_h5"]]
    df = df.merge(h5, on="game_id", how="left")
    df["spread_chg_2d"] = df["consensus_spread_h3"] - df["consensus_spread_h5"]

    df = df.merge(crossings, on="game_id", how="left")
    for k in KEY_NUMBERS:
        df[f"crossed_key_{k}"] = df[f"crossed_key_{k}"].fillna(0).astype(int)

    model1_df = build_model1_dataset()
    context_cols = ["game_id", "home_team", "away_team", "week"] + MODEL1_NUMERIC_FEATURES
    df = df.merge(model1_df[context_cols], on="game_id", how="left")

    oof = compute_model1_oof_predictions(model1_df)
    df = df.merge(oof, on="game_id", how="left")
    df["model1_minus_consensus"] = df["model1_pred_spread"] - df["consensus_spread_h3"]

    required = ["target"] + ALL_NUMERIC
    n_before = len(df)
    df = df.dropna(subset=required)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"Dropped {n_dropped}/{n_before} games with missing target/feature values "
              f"(no horizon snapshot at h3/h5, or no model_1 OOF prediction for the first season)")

    df = df.reset_index(drop=True)
    df["season"] = df["season"].astype(int)

    return df


def _metrics(y_true, y_pred) -> dict:
    return {
        "n": len(y_true),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _print_metrics_table(per_season: dict, pooled: dict, label: str) -> None:
    has_alpha = any("alpha" in m for m in per_season.values())
    header = f"  {'season':>8}  {'n':>5}  {'rmse':>7}  {'mae':>7}  {'r2':>8}"
    if has_alpha:
        header += f"  {'alpha':>8}"
    print(f"\n{label} — walk-forward by season")
    print(header)
    for season, m in sorted(per_season.items()):
        row = f"  {season:>8}  {m['n']:>5}  {m['rmse']:>7.3f}  {m['mae']:>7.3f}  {m['r2']:>8.3f}"
        if has_alpha:
            row += f"  {m['alpha']:>8.3g}"
        print(row)
    footer = f"  {'POOLED':>8}  {pooled['n']:>5}  {pooled['rmse']:>7.3f}  {pooled['mae']:>7.3f}  {pooled['r2']:>8.3f}"
    print(footer)


def walk_forward_ridge(df: pd.DataFrame, numeric_features):
    """Ridge regression, walk-forward by season. Alpha is selected by
    RidgeCV's internal cross-validation on the training fold ONLY — the
    test fold is never touched until after alpha is fixed and the model
    refit on the full training fold with it. Features are standardized
    using scaler statistics fit on the training fold only, too."""
    seasons = sorted(df.season.unique())
    per_season = {}
    all_true, all_pred = [], []

    for season in seasons[1:]:
        train = df[df.season < season]
        test = df[df.season == season]
        if train.empty or test.empty:
            continue

        X_train = train[numeric_features].to_numpy(dtype=float)
        y_train = train["target"].to_numpy(dtype=float)
        X_test = test[numeric_features].to_numpy(dtype=float)
        y_test = test["target"].to_numpy(dtype=float)

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RidgeCV(alphas=RIDGE_ALPHAS)  # default cv=None -> efficient LOOCV on the training fold
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)

        m = _metrics(y_test, pred)
        m["alpha"] = model.alpha_
        per_season[season] = m
        all_true.extend(y_test.tolist())
        all_pred.extend(pred.tolist())

    pooled = _metrics(pd.Series(all_true), pd.Series(all_pred))
    return per_season, pooled


def main() -> None:
    df = build_dataset()
    print(f"Dataset: {len(df)} games, seasons {sorted(df.season.unique())}")

    print("\n" + "=" * 70)
    print("REGRESSION (Ridge, alpha selected on training fold only) — walk-forward by feature group")
    print("=" * 70)

    a_per_season, a_pooled = walk_forward_ridge(df, GROUP_A_NUMERIC)
    _print_metrics_table(a_per_season, a_pooled, f"Group A only ({len(GROUP_A_NUMERIC)} features)")

    b_per_season, b_pooled = walk_forward_ridge(df, GROUP_B_NUMERIC)
    _print_metrics_table(b_per_season, b_pooled, f"Group B only ({len(GROUP_B_NUMERIC)} features)")

    combined_per_season, combined_pooled = walk_forward_ridge(df, ALL_NUMERIC)
    _print_metrics_table(combined_per_season, combined_pooled, f"Combined ({len(ALL_NUMERIC)} features)")

    print("\n" + "=" * 70)
    print("Summary — pooled walk-forward regression metrics")
    print("=" * 70)
    print(f"  Group A only: n={a_pooled['n']}  R^2={a_pooled['r2']:.3f}  RMSE={a_pooled['rmse']:.3f}  MAE={a_pooled['mae']:.3f}")
    print(f"  Group B only: n={b_pooled['n']}  R^2={b_pooled['r2']:.3f}  RMSE={b_pooled['rmse']:.3f}  MAE={b_pooled['mae']:.3f}")
    print(f"  Combined:     n={combined_pooled['n']}  R^2={combined_pooled['r2']:.3f}  RMSE={combined_pooled['rmse']:.3f}  MAE={combined_pooled['mae']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
