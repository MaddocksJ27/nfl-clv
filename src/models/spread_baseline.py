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
from src.features.player_join import normalize_name, resolve_player_name
from src.ingest.bulk_props import game_type_for_week

REPO_ROOT = Path(__file__).resolve().parents[2]
EPA_RATINGS_PATH = REPO_ROOT / "data" / "interim" / "epa_ratings.parquet"
PBP_PATH = REPO_ROOT / "data" / "raw" / "pbp.parquet"
ROSTERS_PATH = REPO_ROOT / "data" / "raw" / "rosters.parquet"
SNAP_COUNTS_PATH = REPO_ROOT / "data" / "raw" / "snap_counts.parquet"
INJURIES_PATH = REPO_ROOT / "data" / "raw" / "injuries.parquet"

EPA_SPLITS = ("overall", "pass", "run")
EPA_COLS = [f"{split}_{side}_epa" for split in EPA_SPLITS for side in ("off", "def")]

BASE_NUMERIC_FEATURES = [f"home_{c}" for c in EPA_COLS] + [f"away_{c}" for c in EPA_COLS] + [
    "home_rest", "away_rest", "div_game", "week",
]
CATEGORICAL_FEATURES = ["roof", "surface"]

QB_WINDOW_DROPBACKS = 200  # "last N dropbacks" for the rolling QB rating
QB_SHRINK_K = 100          # shrinkage weight, in dropbacks, toward the replacement baseline

QB_NUMERIC_FEATURES = ["home_qb_rating", "away_qb_rating", "home_qb_change", "away_qb_change"]
QB_CATEGORICAL_FEATURES = ["home_qb_injury", "away_qb_injury"]

NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + QB_NUMERIC_FEATURES
NO_SPLIT_NUMERIC_FEATURES = [
    f for f in NUMERIC_FEATURES if not (f.startswith("home_pass_") or f.startswith("home_run_")
                                          or f.startswith("away_pass_") or f.startswith("away_run_"))
]
ALL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + QB_CATEGORICAL_FEATURES

LGBM_PARAMS = dict(
    n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1,
)


def identify_actual_starters(snap_counts: pd.DataFrame = None, rosters: pd.DataFrame = None) -> pd.DataFrame:
    """One row per (season, week, team) that actually played: the QB with
    the most offense snaps that game, resolved to a GSIS player_id by
    name-matching against that team-season's roster (reusing the
    player_join resolver — snap_counts only carries a PFR id, incompatible
    with the GSIS ids used everywhere else: pbp's passer_player_id,
    rosters' player_id, injuries' gsis_id)."""
    if snap_counts is None:
        snap_counts = pd.read_parquet(SNAP_COUNTS_PATH)
    if rosters is None:
        rosters = pd.read_parquet(ROSTERS_PATH)

    qb_snaps = snap_counts[snap_counts.position == "QB"].sort_values("offense_snaps", ascending=False)
    starters = qb_snaps.drop_duplicates(subset=["season", "week", "team"], keep="first").copy()

    pool_cache = {}
    resolved_ids = []
    for row in starters.itertuples():
        key = (row.season, row.team)
        if key not in pool_cache:
            pool = rosters[(rosters.season == row.season) & (rosters.team == row.team)][
                ["player_id", "player_name", "position", "team"]].drop_duplicates(subset="player_id").copy()
            pool["normalized_name"] = pool.player_name.map(normalize_name)
            pool_cache[key] = pool
        result = resolve_player_name(row.player, pool_cache[key], position_hint="QB")
        resolved_ids.append(result["player_id"])

    starters["player_id"] = resolved_ids
    n_unresolved = pd.isna(starters["player_id"]).sum()
    if n_unresolved:
        print(f"Starter QB identification: {n_unresolved}/{len(starters)} names unresolved against roster")

    return starters[["season", "week", "team", "player_id"]].reset_index(drop=True)


def build_projected_starters(schedules: pd.DataFrame, actual_starters: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team) for every game in the schedule:
    projected_player_id (the most recently identified actual starter
    strictly before this week — carried forward across byes and season
    boundaries, since that's the only thing knowable in advance) and
    qb_change (1 if that differs from what was projected the previous
    week, else 0; 0 whenever either side is unknown)."""
    home = schedules[["season", "week", "home_team"]].rename(columns={"home_team": "team"})
    away = schedules[["season", "week", "away_team"]].rename(columns={"away_team": "team"})
    idx = pd.concat([home, away], ignore_index=True).drop_duplicates()

    merged = idx.merge(actual_starters, on=["season", "week", "team"], how="left")
    merged = merged.sort_values(["team", "season", "week"]).reset_index(drop=True)

    ffilled = merged.groupby("team")["player_id"].ffill()
    merged["projected_player_id"] = ffilled.groupby(merged["team"]).shift(1)

    prev_projected = merged.groupby("team")["projected_player_id"].shift(1)
    both_known = prev_projected.notna() & merged["projected_player_id"].notna()
    merged["qb_change"] = ((prev_projected != merged["projected_player_id"]) & both_known).astype(int)

    return merged[["season", "week", "team", "projected_player_id", "qb_change"]]


def load_qb_dropback_log() -> pd.DataFrame:
    """One row per (season, week, team, qb_id): n_dropbacks, sum_epa.
    qb_dropback plays include scrambles (play_type='run', no passer —
    identified there is the QB via rusher_player_id instead)."""
    cols = ["season", "week", "posteam", "play_type", "qb_dropback", "epa",
            "two_point_attempt", "passer_player_id", "rusher_player_id"]
    pbp = pd.read_parquet(PBP_PATH, columns=cols)
    db = pbp[
        (pbp.qb_dropback == 1) & pbp.epa.notna() & (pbp.two_point_attempt.fillna(0) != 1) & pbp.posteam.notna()
    ].copy()
    db["qb_id"] = db["passer_player_id"].fillna(db["rusher_player_id"])
    db = db.dropna(subset=["qb_id"])

    log = db.groupby(["season", "week", "posteam", "qb_id"]).epa.agg(
        n_dropbacks="count", sum_epa="sum").reset_index().rename(columns={"posteam": "team"})
    return log


def _replacement_baseline_asof(qb_log: pd.DataFrame, actual_starters: pd.DataFrame, season: int, week: int) -> float:
    """Mean EPA/dropback among BACKUP usage (a passer who did not record
    that game's most offense snaps at QB) strictly before (season, week) —
    a genuine "replacement level" proxy, distinct from the full-league mean
    which is dominated by entrenched starters."""
    before = qb_log[(qb_log.season < season) | ((qb_log.season == season) & (qb_log.week < week))]
    if before.empty:
        return 0.0
    starter_lookup = actual_starters.set_index(["season", "week", "team"])["player_id"]
    keys = list(zip(before.season, before.week, before.team))
    starter_ids = starter_lookup.reindex(keys).to_numpy()
    is_backup = before.qb_id.to_numpy() != starter_ids  # NaN starter (unresolved) -> treated as backup by default
    backup = before[is_backup]
    n = backup.n_dropbacks.sum()
    if n == 0:
        return 0.0
    return float(backup.sum_epa.sum() / n)


def _qb_rolling_rating(qb_hist: pd.DataFrame, season: int, week: int, replacement: float,
                        window: int = QB_WINDOW_DROPBACKS, k: float = QB_SHRINK_K) -> float:
    """qb_hist: one QB's own (season, week, n_dropbacks, sum_epa) rows,
    league-wide (any team). Shrinks his last `window` dropbacks' own
    EPA/dropback toward `replacement`, weight = n_used / (n_used + k)."""
    hist = qb_hist[(qb_hist.season < season) | ((qb_hist.season == season) & (qb_hist.week < week))]
    if hist.empty:
        return replacement
    hist = hist.sort_values(["season", "week"], ascending=False)

    cum = hist["n_dropbacks"].cumsum()
    full_mask = (cum <= window).to_numpy()
    n_full = hist.loc[full_mask, "n_dropbacks"].sum()
    sum_full = hist.loc[full_mask, "sum_epa"].sum()

    n_full_rows = int(full_mask.sum())
    remaining = window - n_full
    if remaining > 0 and n_full_rows < len(hist):
        next_row = hist.iloc[n_full_rows]
        frac = remaining / next_row["n_dropbacks"]
        n_full += remaining
        sum_full += next_row["sum_epa"] * frac

    if n_full <= 0:
        return replacement
    raw_mean = sum_full / n_full
    shrink_w = n_full / (n_full + k)
    return replacement + shrink_w * (raw_mean - replacement)


def build_qb_features(games: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """games: one row per matchup with season, week, home_team, away_team
    (as in the modeling dataset). Returns the same rows with home_/away_
    qb_rating, qb_change, qb_injury columns added."""
    actual_starters = identify_actual_starters()
    projected = build_projected_starters(schedules, actual_starters)
    qb_log = load_qb_dropback_log()
    qb_log_by_id = {qb_id: g for qb_id, g in qb_log.groupby("qb_id")}

    injuries = pd.read_parquet(INJURIES_PATH, columns=["season", "week", "team", "gsis_id", "report_status", "date_modified"])
    injuries = injuries.sort_values("date_modified").drop_duplicates(subset=["season", "week", "team", "gsis_id"], keep="last")

    home = projected.rename(columns={"team": "home_team", "projected_player_id": "home_qb_id", "qb_change": "home_qb_change"})
    away = projected.rename(columns={"team": "away_team", "projected_player_id": "away_qb_id", "qb_change": "away_qb_change"})
    out = games.merge(home, on=["season", "week", "home_team"], how="left")
    out = out.merge(away, on=["season", "week", "away_team"], how="left")

    replacement_cache = {}
    home_ratings, away_ratings = [], []
    for row in out.itertuples():
        key = (row.season, row.week)
        if key not in replacement_cache:
            replacement_cache[key] = _replacement_baseline_asof(qb_log, actual_starters, row.season, row.week)
        replacement = replacement_cache[key]

        home_hist = qb_log_by_id.get(row.home_qb_id)
        away_hist = qb_log_by_id.get(row.away_qb_id)
        home_ratings.append(_qb_rolling_rating(home_hist, row.season, row.week, replacement) if home_hist is not None else replacement)
        away_ratings.append(_qb_rolling_rating(away_hist, row.season, row.week, replacement) if away_hist is not None else replacement)

    out["home_qb_rating"] = home_ratings
    out["away_qb_rating"] = away_ratings

    inj_home = injuries.rename(columns={"team": "home_team", "gsis_id": "home_qb_id", "report_status": "home_qb_injury"})
    inj_away = injuries.rename(columns={"team": "away_team", "gsis_id": "away_qb_id", "report_status": "away_qb_injury"})
    out = out.merge(inj_home[["season", "week", "home_team", "home_qb_id", "home_qb_injury"]],
                     on=["season", "week", "home_team", "home_qb_id"], how="left")
    out = out.merge(inj_away[["season", "week", "away_team", "away_qb_id", "away_qb_injury"]],
                     on=["season", "week", "away_team", "away_qb_id"], how="left")
    out["home_qb_injury"] = out["home_qb_injury"].fillna("Healthy").replace("None", "Healthy")
    out["away_qb_injury"] = out["away_qb_injury"].fillna("Healthy").replace("None", "Healthy")
    out["home_qb_change"] = out["home_qb_change"].fillna(0).astype(int)
    out["away_qb_change"] = out["away_qb_change"].fillna(0).astype(int)

    return out


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

    df = build_qb_features(df, schedules)

    required = ["target_spread"] + NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES
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
    for col in ALL_CATEGORICAL_FEATURES:
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


def _ols_formula(numeric_features, categorical_features) -> str:
    terms = numeric_features + [f"C({c})" for c in categorical_features]
    return "target_spread ~ " + " + ".join(terms)


def fit_full_ols(df: pd.DataFrame, numeric_features=NUMERIC_FEATURES, categorical_features=ALL_CATEGORICAL_FEATURES):
    """OLS on the full dataset, purely for a readable coefficient table —
    not used for the walk-forward metrics (see walk_forward_ols)."""
    return smf.ols(_ols_formula(numeric_features, categorical_features), data=df).fit()


def walk_forward_ols(df: pd.DataFrame, numeric_features=NUMERIC_FEATURES, categorical_features=ALL_CATEGORICAL_FEATURES):
    formula = _ols_formula(numeric_features, categorical_features)
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


def walk_forward_lgbm(df: pd.DataFrame, numeric_features=NUMERIC_FEATURES, categorical_features=ALL_CATEGORICAL_FEATURES):
    dummies = pd.get_dummies(df[categorical_features], drop_first=False)
    X_full = pd.concat([df[numeric_features], dummies], axis=1)
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
    print("OLS coefficients — WITH pass_/run_ splits (fit on all available games)")
    print("=" * 70)
    print(full_model.summary())

    ols_per_season, ols_pooled = walk_forward_ols(df)
    _print_metrics_table(ols_per_season, ols_pooled, "OLS (with pass_/run_ splits)")

    lgbm_per_season, lgbm_pooled = walk_forward_lgbm(df)
    _print_metrics_table(lgbm_per_season, lgbm_pooled, "LightGBM (with pass_/run_ splits)")

    no_split_model = fit_full_ols(df, numeric_features=NO_SPLIT_NUMERIC_FEATURES)
    print("\n" + "=" * 70)
    print("OLS coefficients — WITHOUT pass_/run_ splits (fit on all available games)")
    print("=" * 70)
    print(no_split_model.summary())

    ols_ns_per_season, ols_ns_pooled = walk_forward_ols(df, numeric_features=NO_SPLIT_NUMERIC_FEATURES)
    _print_metrics_table(ols_ns_per_season, ols_ns_pooled, "OLS (no pass_/run_ splits)")

    print("\n" + "=" * 70)
    print("Summary — pooled walk-forward metrics")
    print("=" * 70)
    print(f"  OLS, with splits:    R^2={ols_pooled['r2']:.3f}  RMSE={ols_pooled['rmse']:.3f}  MAE={ols_pooled['mae']:.3f}")
    print(f"  OLS, no splits:      R^2={ols_ns_pooled['r2']:.3f}  RMSE={ols_ns_pooled['rmse']:.3f}  MAE={ols_ns_pooled['mae']:.3f}")
    print(f"  LightGBM, with splits: R^2={lgbm_pooled['r2']:.3f}  RMSE={lgbm_pooled['rmse']:.3f}  MAE={lgbm_pooled['mae']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
