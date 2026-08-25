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

Features:
  GROUP A — state at the horizon:
    consensus spread/total at kickoff-3d, book disagreement (SD of spread
    across books) at kickoff-3d, model_1's predicted CLOSE spread minus
    the kickoff-3d consensus (how far the market sits from the fundamental
    estimate), EPA ratings + QB features as-of that week, week number,
    rest days each side, div_game, roof.
  GROUP B — movement already underway at the horizon:
    spread change over the 7 days and 2 days before the horizon, whether
    the line crossed a key number (3 or 7) in the 7 days before the horizon.

model_1 (src.models.spread_baseline's recommended no-split OLS) is refit
walk-forward with the SAME season folds used here, so its predictions
never see a season this module is being evaluated on either.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.eval.validate_featured import (
    build_event_schedule_map, compute_close_spreads, compute_consensus_home_spreads,
    compute_fixed_horizon_open, load_data,
)
from src.models.spread_baseline import ALL_CATEGORICAL_FEATURES as MODEL1_CATEGORICAL_FEATURES
from src.models.spread_baseline import NO_SPLIT_NUMERIC_FEATURES as MODEL1_NUMERIC_FEATURES
from src.models.spread_baseline import build_dataset as build_model1_dataset

HORIZON_DAYS = 3
WEEK_BEFORE_DAYS = 7   # for the 7-day-before-horizon change and key-number-crossing window
SHORT_BEFORE_DAYS = 2  # for the 2-day-before-horizon change
KEY_NUMBERS = (3, 7)

GROUP_A_NUMERIC = [
    "consensus_spread_h3", "consensus_total_h3", "book_disagreement_h3", "model1_minus_consensus",
] + MODEL1_NUMERIC_FEATURES  # EPA ratings + QB features + week/home_rest/away_rest/div_game, as-of that week
GROUP_A_CATEGORICAL = ["roof"]

GROUP_B_NUMERIC = ["spread_chg_7d", "spread_chg_2d", "crossed_key_3", "crossed_key_7"]
GROUP_B_CATEGORICAL = []

ALL_NUMERIC = GROUP_A_NUMERIC + GROUP_B_NUMERIC
ALL_CATEGORICAL = GROUP_A_CATEGORICAL + GROUP_B_CATEGORICAL

DRIFT_CLASS_EDGES = [-np.inf, -1, 1, np.inf]
DRIFT_CLASS_LABELS = ["home_gained", "flat", "away_gained"]


def compute_consensus_totals(featured: pd.DataFrame, event_map: pd.DataFrame) -> pd.DataFrame:
    """game_id, snapshot_time, consensus_total — mean of the `line` across
    books at each snapshot for the totals market (Over/Under rows carry
    the same point value, so this just averages across books)."""
    totals = featured[featured.market == "totals"]
    game_lookup = event_map.set_index("event_id")["game_id"]
    tagged = totals[totals.event_id.isin(game_lookup.index)].copy()
    tagged["game_id"] = tagged.event_id.map(game_lookup)

    consensus = tagged.groupby(["game_id", "snapshot_time"]).line.mean().reset_index()
    consensus.columns = ["game_id", "snapshot_time", "consensus_total"]
    consensus["snapshot_ts"] = pd.to_datetime(consensus.snapshot_time, utc=True)
    return consensus


def compute_fixed_horizon_total(consensus_totals: pd.DataFrame, games: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    merged = consensus_totals.merge(games[["game_id", "commence_ts"]], on="game_id")
    cutoff = merged.commence_ts - pd.Timedelta(days=horizon_days)
    eligible = merged[merged.snapshot_ts <= cutoff]
    latest = eligible.sort_values("snapshot_ts").groupby("game_id").tail(1).copy()
    return latest.rename(columns={"consensus_total": "open_total"})[["game_id", "open_total"]]


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
    min/max within (kickoff-10d, kickoff-3d] straddle +-3 / +-7, else 0 —
    a same-day-granularity proxy for "did the line cross this key number
    at any point in the week before the horizon" (can't see intra-day
    crossings the daily snapshot cadence didn't capture)."""
    merged = consensus.merge(games[["game_id", "commence_ts"]], on="game_id")
    upper = merged.commence_ts - pd.Timedelta(days=HORIZON_DAYS)
    lower = upper - pd.Timedelta(days=WEEK_BEFORE_DAYS)
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
    Group A and Group B features, all knowable at kickoff-3d."""
    featured, schedules = load_data()
    event_map = build_event_schedule_map(featured, schedules)
    games = event_map.drop_duplicates(subset="game_id")

    consensus = compute_consensus_home_spreads(featured, event_map)
    close = compute_close_spreads(consensus, games)
    horizon3 = compute_fixed_horizon_open(consensus, games, HORIZON_DAYS)
    horizon5 = compute_fixed_horizon_open(consensus, games, HORIZON_DAYS + SHORT_BEFORE_DAYS)
    horizon10 = compute_fixed_horizon_open(consensus, games, HORIZON_DAYS + WEEK_BEFORE_DAYS)

    consensus_totals = compute_consensus_totals(featured, event_map)
    total_h3 = compute_fixed_horizon_total(consensus_totals, games, HORIZON_DAYS)

    disagreement = compute_book_disagreement(featured, event_map, horizon3)
    crossings = compute_key_number_crossings(consensus, games)

    df = close.merge(horizon3, on=["game_id", "season"]).rename(columns={"open_spread": "consensus_spread_h3"})
    df = df.merge(total_h3, on="game_id").rename(columns={"open_total": "consensus_total_h3"})
    df["target"] = df["close_spread"] - df["consensus_spread_h3"]

    df = df.merge(disagreement, on="game_id", how="left")

    h5 = horizon5.rename(columns={"open_spread": "consensus_spread_h5"})[["game_id", "consensus_spread_h5"]]
    h10 = horizon10.rename(columns={"open_spread": "consensus_spread_h10"})[["game_id", "consensus_spread_h10"]]
    df = df.merge(h5, on="game_id", how="left").merge(h10, on="game_id", how="left")
    df["spread_chg_2d"] = df["consensus_spread_h3"] - df["consensus_spread_h5"]
    df["spread_chg_7d"] = df["consensus_spread_h3"] - df["consensus_spread_h10"]

    df = df.merge(crossings, on="game_id", how="left")
    for k in KEY_NUMBERS:
        df[f"crossed_key_{k}"] = df[f"crossed_key_{k}"].fillna(0).astype(int)

    model1_df = build_model1_dataset()
    # MODEL1_NUMERIC_FEATURES already includes week/home_rest/away_rest/div_game
    context_cols = ["game_id", "home_team", "away_team", "roof"] + MODEL1_NUMERIC_FEATURES
    df = df.merge(model1_df[context_cols], on="game_id", how="left")

    oof = compute_model1_oof_predictions(model1_df)
    df = df.merge(oof, on="game_id", how="left")
    df["model1_minus_consensus"] = df["model1_pred_spread"] - df["consensus_spread_h3"]

    required = ["target"] + ALL_NUMERIC + ALL_CATEGORICAL
    n_before = len(df)
    df["roof"] = df["roof"].fillna("unknown")
    df = df.dropna(subset=required)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"Dropped {n_dropped}/{n_before} games with missing target/feature values "
              f"(no horizon snapshot at h3/h5/h10, or no model_1 OOF prediction for the first season)")

    df = df.reset_index(drop=True)
    df["season"] = df["season"].astype(int)
    for col in ALL_CATEGORICAL:
        df[col] = pd.Categorical(df[col], categories=sorted(df[col].unique()))

    df["drift_class"] = pd.cut(df["target"], bins=DRIFT_CLASS_EDGES, labels=DRIFT_CLASS_LABELS)

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


def walk_forward_ols(df: pd.DataFrame, numeric_features, categorical_features):
    formula = _formula(numeric_features, categorical_features)
    seasons = sorted(df.season.unique())
    per_season = {}
    all_true, all_pred = [], []

    for season in seasons[1:]:
        train = df[df.season < season]
        test = df[df.season == season]
        if train.empty or test.empty:
            continue
        model = smf.ols(formula, data=train).fit()
        pred = model.predict(test)
        per_season[season] = _metrics(test.target, pred)
        all_true.extend(test.target.tolist())
        all_pred.extend(pred.tolist())

    pooled = _metrics(pd.Series(all_true), pd.Series(all_pred))
    return per_season, pooled


def walk_forward_classification(df: pd.DataFrame, numeric_features, categorical_features):
    dummies = pd.get_dummies(df[categorical_features], drop_first=True) if categorical_features else pd.DataFrame(index=df.index)
    X_full = pd.concat([df[numeric_features], dummies], axis=1)
    y_full = df["drift_class"].astype(str)

    seasons = sorted(df.season.unique())
    per_season_acc = {}
    all_true, all_pred = [], []

    for season in seasons[1:]:
        train_mask = (df.season < season).to_numpy()
        test_mask = (df.season == season).to_numpy()
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_full[train_mask])
        X_test = scaler.transform(X_full[test_mask])

        clf = LogisticRegression(max_iter=2000, C=1.0)  # lbfgs solver is multinomial by default since sklearn 1.5+
        clf.fit(X_train, y_full[train_mask])
        pred = clf.predict(X_test)

        per_season_acc[season] = accuracy_score(y_full[test_mask], pred)
        all_true.extend(y_full[test_mask].tolist())
        all_pred.extend(pred.tolist())

    overall_acc = accuracy_score(all_true, all_pred)
    cm = confusion_matrix(all_true, all_pred, labels=DRIFT_CLASS_LABELS)
    return per_season_acc, overall_acc, cm, pd.Series(all_true)


def main() -> None:
    df = build_dataset()
    print(f"Dataset: {len(df)} games, seasons {sorted(df.season.unique())}")

    print("\n" + "=" * 70)
    print("REGRESSION — walk-forward pooled R^2 by feature group")
    print("=" * 70)

    a_per_season, a_pooled = walk_forward_ols(df, GROUP_A_NUMERIC, GROUP_A_CATEGORICAL)
    _print_metrics_table(a_per_season, a_pooled, "Group A only")

    b_per_season, b_pooled = walk_forward_ols(df, GROUP_B_NUMERIC, GROUP_B_CATEGORICAL)
    _print_metrics_table(b_per_season, b_pooled, "Group B only")

    combined_per_season, combined_pooled = walk_forward_ols(df, ALL_NUMERIC, ALL_CATEGORICAL)
    _print_metrics_table(combined_per_season, combined_pooled, "Combined (A + B)")

    print("\n" + "=" * 70)
    print("Summary — pooled walk-forward regression metrics")
    print("=" * 70)
    print(f"  Group A only: R^2={a_pooled['r2']:.3f}  RMSE={a_pooled['rmse']:.3f}  MAE={a_pooled['mae']:.3f}")
    print(f"  Group B only: R^2={b_pooled['r2']:.3f}  RMSE={b_pooled['rmse']:.3f}  MAE={b_pooled['mae']:.3f}")
    print(f"  Combined:     R^2={combined_pooled['r2']:.3f}  RMSE={combined_pooled['rmse']:.3f}  MAE={combined_pooled['mae']:.3f}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("CLASSIFICATION — drift < -1 / |drift| <= 1 / drift > +1, combined features")
    print("=" * 70)
    per_season_acc, overall_acc, cm, all_true = walk_forward_classification(df, ALL_NUMERIC, ALL_CATEGORICAL)

    print("\nAccuracy by season:")
    for season, acc in sorted(per_season_acc.items()):
        print(f"  {season}: {acc:.3f}")
    print(f"  POOLED: {overall_acc:.3f}")

    print("\nConfusion matrix (rows=true, cols=predicted), labels =", DRIFT_CLASS_LABELS)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in DRIFT_CLASS_LABELS],
                          columns=[f"pred_{l}" for l in DRIFT_CLASS_LABELS])
    print(cm_df.to_string())

    print("\nClass balance (true labels, all seasons pooled):")
    balance = all_true.value_counts()
    balance_pct = (all_true.value_counts(normalize=True) * 100).round(1)
    for label in DRIFT_CLASS_LABELS:
        print(f"  {label}: {balance.get(label, 0)} ({balance_pct.get(label, 0.0)}%)")

    majority_baseline = balance.max() / len(all_true)
    print(f"\nMajority-class baseline (always predict '{balance.idxmax()}'): {majority_baseline:.3f}")
    print(f"Model pooled accuracy:                                        {overall_acc:.3f}")
    if overall_acc <= majority_baseline:
        print("-> Model does NOT beat the trivial majority-class baseline.")


if __name__ == "__main__":
    main()
