"""Opponent-adjusted, recency-weighted team EPA/play ratings, computed
as-of each week using only games played strictly before it.

Six ratings per (season, week, team): offensive/defensive EPA per play for
dropbacks (pass), rush, and overall.

## Method

1. Filter plays to epa not null, play_type in (pass, run), excluding
   two-point attempts (kneels/spikes already have their own play_type and
   are excluded by the pass/run filter alone).
2. Collapse to one row per (season, week, game, team, opponent) — offensive
   n_plays/sum_epa for overall/pass/run. A team's defensive numbers for a
   game are just the opponent's offensive row for that same game.
3. For each team, within a season, order real games most-recent-first and
   weight game i by 0.5**(games_back_i / half_life_games). Games_back is
   computed separately for a team's offensive and defensive game order
   (they're the same games, but "how many games back" is always relative
   to that team's own schedule, not the opponent's).
4. Opponent adjustment: iterate — each team's offensive rating is the
   recency-weighted mean of (play epa - opponent's current defensive
   deviation), and vice versa for defense — until the largest single-team
   change is below tolerance or 20 iterations are used.
5. Season transitions: week 1 of a season (other than the first season
   present in the data) starts from carry_fraction * that team's "final"
   rating from the prior season (computed the same way, as of one week
   past the prior season's last week), treated as a single recency-weighted
   pseudo-observation positioned immediately before the season's actual
   games — so as real games accumulate through the season, the carry-in
   naturally fades out via the same half-life decay. Cross-season history
   otherwise does NOT blend via decay directly; the carry-in is the only
   link between seasons, by design (a clean, explicit reset that doesn't
   depend on exactly how much old data survives decay).
6. Shrinkage: each team's blended deviation from the league mean is scaled
   by effective_games / (effective_games + shrink_k), where effective_games
   is the count of real games contributing plus 1 for a present carry-in.
7. The league mean (mu) is recomputed per (season, week) boundary from all
   qualifying plays strictly before it, across all prior seasons (a stable,
   slowly-moving reference, unlike the team-specific ratings).

Team abbreviations are normalized via team_names.normalize_team_abbr before
anything else, so a relocated franchise's history stays continuous (moot
for the cached pbp data specifically, which nflverse already backfills to
current abbreviations, but applied unconditionally as defensive handling).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest.team_names import normalize_team_abbr, team_name_map

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PBP_PATH = REPO_ROOT / "data" / "raw" / "pbp.parquet"
OUTPUT_PATH = REPO_ROOT / "data" / "interim" / "epa_ratings.parquet"

SEASONS = list(range(2016, 2026))
SPLITS = ("overall", "pass", "run")

DEFAULT_HALF_LIFE_GAMES = 10.0
DEFAULT_SHRINK_K = 4.0
DEFAULT_CARRY_FRACTION = 0.35
MAX_ITER = 500
CONV_TOL = 1e-4  # team EPA/play ratings span roughly +/-0.3; 1e-4 is ~0.03% of that range
DAMPING = 0.5  # under-relaxation factor; prevents the offense/defense mutual update from oscillating

ALL_TEAMS = sorted(set(team_name_map().values()))

PBP_COLUMNS = ["season", "week", "game_id", "play_type", "epa", "posteam", "defteam", "two_point_attempt"]


def load_plays(raw: pd.DataFrame = None) -> pd.DataFrame:
    """Apply the play-level filter (epa not null, play_type in pass/run,
    exclude two-point attempts) and normalize team abbreviations. If `raw`
    is None, reads the cached pbp parquet; callers (notably tests) may pass
    an already-loaded/pre-restricted dataframe instead."""
    if raw is None:
        raw = pd.read_parquet(PBP_PATH, columns=PBP_COLUMNS)

    df = raw[
        raw.epa.notna()
        & raw.play_type.isin(["pass", "run"])
        & (raw.two_point_attempt.fillna(0) != 1)
        & raw.posteam.notna()
        & raw.defteam.notna()
    ].copy()
    df["posteam"] = normalize_team_abbr(df["posteam"])
    df["defteam"] = normalize_team_abbr(df["defteam"])
    return df


def build_team_games(plays: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, game_id, team, opponent): offensive
    n_plays_<split>/sum_epa_<split> for split in (overall, pass, run).
    A team's defensive numbers for a game are the row where opponent==team."""
    base = plays.drop_duplicates(subset=["season", "week", "game_id", "posteam", "defteam"])[
        ["season", "week", "game_id", "posteam", "defteam"]
    ].rename(columns={"posteam": "team", "defteam": "opponent"})

    split_plays = {
        "overall": plays,
        "pass": plays[plays.play_type == "pass"],
        "run": plays[plays.play_type == "run"],
    }
    for split, sub in split_plays.items():
        g = sub.groupby(["season", "week", "game_id", "posteam", "defteam"]).epa.agg(
            **{f"n_plays_{split}": "count", f"sum_epa_{split}": "sum"}
        ).reset_index().rename(columns={"posteam": "team", "defteam": "opponent"})
        base = base.merge(g, on=["season", "week", "game_id", "team", "opponent"], how="left")

    for split in split_plays:
        base[f"n_plays_{split}"] = base[f"n_plays_{split}"].fillna(0).astype(int)
        base[f"sum_epa_{split}"] = base[f"sum_epa_{split}"].fillna(0.0)

    return base


def _mu_asof(team_games: pd.DataFrame, split: str, season: int, week: int) -> float:
    """League-mean EPA/play for `split`, over all qualifying plays strictly
    before (season, week), across all seasons. 0.0 if no such plays exist
    (only possible at the very first boundary in the whole dataset)."""
    before = team_games[(team_games.season < season) | ((team_games.season == season) & (team_games.week < week))]
    n = before[f"n_plays_{split}"].sum()
    if n == 0:
        return 0.0
    return before[f"sum_epa_{split}"].sum() / n


def _opponent_adjust(real_games: pd.DataFrame, split: str, mu: float, half_life: float, max_iter: int = MAX_ITER):
    """real_games: this team-season's real games (already restricted to
    strictly-earlier weeks), one row per (team, opponent, week, n_plays,
    sum_epa) for this split. Returns (off_dev, def_dev, off_weight_sum,
    def_weight_sum, iterations_used, final_max_change) — the dev dicts are
    deviations from mu, the weight sums are each team's total recency
    weight (used later to blend against a season carry-in), and
    final_max_change is the largest single-team change made on the last
    iteration performed (diagnostic: how far from the convergence
    tolerance a capped-out boundary actually was)."""
    n_col, s_col = f"n_plays_{split}", f"sum_epa_{split}"
    g = real_games[real_games[n_col] > 0].copy()
    if g.empty:
        return {}, {}, {}, {}, 0, 0.0

    g["off_gb"] = g.sort_values(["team", "week"], ascending=[True, False]).groupby("team").cumcount()
    g["def_gb"] = g.sort_values(["opponent", "week"], ascending=[True, False]).groupby("opponent").cumcount()
    g["off_w"] = 0.5 ** (g["off_gb"] / half_life)
    g["def_w"] = 0.5 ** (g["def_gb"] / half_life)

    teams = sorted(set(g.team) | set(g.opponent))
    off_dev = {t: 0.0 for t in teams}
    def_dev = {t: 0.0 for t in teams}

    iterations_used = 0
    max_change = 0.0
    for it in range(1, max_iter + 1):
        iterations_used = it

        # Gauss-Seidel + damping: naive simultaneous (Jacobi) updates of two
        # mutually-dependent series like this one oscillate indefinitely
        # rather than converging (verified empirically — offense and defense
        # bounce between two values forever without damping). Using the
        # just-updated offense values for the same iteration's defense
        # update, and under-relaxing each step by DAMPING, converges smoothly.
        opp_def = g["opponent"].map(def_dev).fillna(0.0)
        adj_off = g[s_col] - g[n_col] * opp_def
        num_off = (g["off_w"] * adj_off).groupby(g["team"]).sum()
        den_off = (g["off_w"] * g[n_col]).groupby(g["team"]).sum()
        computed_off = (num_off / den_off) - mu
        new_off_d = computed_off.reindex(teams).fillna(0.0).to_dict()
        new_off_d = {t: off_dev[t] + DAMPING * (new_off_d[t] - off_dev[t]) for t in teams}

        opp_off_dev = g["team"].map(new_off_d).fillna(0.0)
        adj_def = g[s_col] - g[n_col] * opp_off_dev
        num_def = (g["def_w"] * adj_def).groupby(g["opponent"]).sum()
        den_def = (g["def_w"] * g[n_col]).groupby(g["opponent"]).sum()
        computed_def = (num_def / den_def) - mu
        new_def_d = computed_def.reindex(teams).fillna(0.0).to_dict()
        new_def_d = {t: def_dev[t] + DAMPING * (new_def_d[t] - def_dev[t]) for t in teams}

        max_change = max(
            max(abs(new_off_d[t] - off_dev[t]) for t in teams),
            max(abs(new_def_d[t] - def_dev[t]) for t in teams),
        )
        off_dev, def_dev = new_off_d, new_def_d
        if max_change < CONV_TOL:
            break

    off_weight_sum = g.groupby("team")["off_w"].sum().to_dict()
    def_weight_sum = g.groupby("opponent")["def_w"].sum().to_dict()
    return off_dev, def_dev, off_weight_sum, def_weight_sum, iterations_used, max_change


def _rating_for_boundary(team_games: pd.DataFrame, season: int, week: int, carry_in,
                          half_life: float, shrink_k: float) -> dict:
    """Ratings for every team as-of (season, week), for all three splits.
    carry_in: {team: {split: {"off": dev, "def": dev}}} from the prior
    season's final rating (already scaled by carry_fraction), or None if
    `season` is the first season present in team_games.

    Returns {team: {split: {"off": epa_per_play, "def": epa_per_play}}}
    plus a parallel {split: iterations_used} dict under key "_iterations".
    """
    real_games = team_games[(team_games.season == season) & (team_games.week < week)]
    n_real_games = real_games.groupby("team").size().to_dict()  # per-team count, same across splits

    result = {t: {} for t in ALL_TEAMS}
    iterations = {}

    for split in SPLITS:
        mu = _mu_asof(team_games, split, season, week)
        off_dev, def_dev, off_w_sum, def_w_sum, iters, _ = _opponent_adjust(real_games, split, mu, half_life)
        iterations[split] = iters

        for t in ALL_TEAMS:
            n_games = n_real_games.get(t, 0)
            real_off = off_dev.get(t)
            real_def = def_dev.get(t)
            real_off_w = off_w_sum.get(t, 0.0)
            real_def_w = def_w_sum.get(t, 0.0)

            carry_off = carry_in[t][split]["off"] if carry_in and t in carry_in else None
            carry_def = carry_in[t][split]["def"] if carry_in and t in carry_in else None
            # positioned as the game immediately before this season's history,
            # so its weight decays the same way as any other game would.
            carry_w = 0.5 ** (n_games / half_life) if carry_in and t in carry_in else 0.0

            def _blend(real_val, real_weight, carry_val):
                parts = []
                if real_val is not None:
                    parts.append((real_val, real_weight))
                if carry_val is not None:
                    parts.append((carry_val, carry_w))
                if not parts:
                    return 0.0
                total_w = sum(w for _, w in parts)
                if total_w == 0:
                    return 0.0
                return sum(v * w for v, w in parts) / total_w

            blended_off = _blend(real_off, real_off_w, carry_off)
            blended_def = _blend(real_def, real_def_w, carry_def)

            effective_games = n_games + (1 if (carry_in and t in carry_in) else 0)
            shrink_w = effective_games / (effective_games + shrink_k)

            result[t][split] = {
                "off": mu + blended_off * shrink_w,
                "def": mu + blended_def * shrink_w,
            }

    result["_iterations"] = iterations
    return result


def _season_weeks(team_games: pd.DataFrame, season: int) -> list:
    return sorted(team_games.loc[team_games.season == season, "week"].unique().tolist())


def compute_all_ratings(plays: pd.DataFrame = None, half_life_games: float = DEFAULT_HALF_LIFE_GAMES,
                         shrink_k: float = DEFAULT_SHRINK_K, carry_fraction: float = DEFAULT_CARRY_FRACTION):
    """Full pipeline: one row per (season, week, team) with all six
    ratings as they stood before that week's games. Returns
    (ratings_df, iteration_log) — iteration_log is a list of
    {season, week, split, iterations} dicts for reporting.
    """
    if plays is None:
        plays = load_plays()
    team_games = build_team_games(plays)

    seasons_present = sorted(team_games.season.unique().tolist())

    rows = []
    iteration_log = []
    final_rating_by_season = {}
    carry_in = None

    for season in seasons_present:
        weeks = _season_weeks(team_games, season)
        for week in weeks:
            r = _rating_for_boundary(team_games, season, week, carry_in, half_life_games, shrink_k)
            for split, iters in r["_iterations"].items():
                iteration_log.append({"season": season, "week": week, "split": split, "iterations": iters})
            for t in ALL_TEAMS:
                row = {"season": season, "week": week, "team": t}
                for split in SPLITS:
                    row[f"{split}_off_epa"] = r[t][split]["off"]
                    row[f"{split}_def_epa"] = r[t][split]["def"]
                rows.append(row)

        # "final" rating for this season = as of one week past its last week,
        # scaled by carry_fraction, becomes next season's carry-in.
        final_week = weeks[-1] + 1
        final = _rating_for_boundary(team_games, season, final_week, carry_in, half_life_games, shrink_k)
        final_rating_by_season[season] = {
            t: {split: {"off": carry_fraction * final[t][split]["off"],
                        "def": carry_fraction * final[t][split]["def"]}
                for split in SPLITS}
            for t in ALL_TEAMS
        }
        carry_in = final_rating_by_season[season]

    ratings_df = pd.DataFrame(rows).sort_values(["season", "week", "team"]).reset_index(drop=True)
    return ratings_df, iteration_log


def compute_ratings_asof(plays_or_raw: pd.DataFrame, target_season: int, target_week: int,
                          half_life_games: float = DEFAULT_HALF_LIFE_GAMES,
                          shrink_k: float = DEFAULT_SHRINK_K,
                          carry_fraction: float = DEFAULT_CARRY_FRACTION,
                          already_filtered: bool = False) -> dict:
    """Standalone, from-scratch computation of every team's rating as-of
    (target_season, target_week), given only a play-level dataframe. Used
    both to keep the pipeline and tests on one code path, and directly by
    tests to verify no-lookahead: pass a dataframe pre-restricted to
    (season, week) strictly earlier than the target and this still
    reproduces the pipeline's stored value for that boundary.

    `already_filtered`: skip load_plays' filtering/normalization (the input
    is already a filtered plays dataframe, e.g. from load_plays() upstream).
    Internally re-restricts to strictly-before (target_season, target_week)
    regardless of what's in the input, as a defensive no-lookahead guard.
    """
    plays = plays_or_raw if already_filtered else load_plays(plays_or_raw)
    plays = plays[(plays.season < target_season) | ((plays.season == target_season) & (plays.week < target_week))]

    team_games = build_team_games(plays)
    seasons_present = sorted(team_games.season.unique().tolist())

    final_rating_by_season = {}
    carry_in = None
    for season in seasons_present:
        if season >= target_season:
            break
        weeks = _season_weeks(team_games, season)
        final_week = weeks[-1] + 1
        final = _rating_for_boundary(team_games, season, final_week, carry_in, half_life_games, shrink_k)
        final_rating_by_season[season] = {
            t: {split: {"off": carry_fraction * final[t][split]["off"],
                        "def": carry_fraction * final[t][split]["def"]}
                for split in SPLITS}
            for t in ALL_TEAMS
        }
        carry_in = final_rating_by_season[season]

    return _rating_for_boundary(team_games, target_season, target_week, carry_in, half_life_games, shrink_k)


def main(half_life_games: float = DEFAULT_HALF_LIFE_GAMES, shrink_k: float = DEFAULT_SHRINK_K,
         carry_fraction: float = DEFAULT_CARRY_FRACTION) -> pd.DataFrame:
    ratings_df, iteration_log = compute_all_ratings(
        half_life_games=half_life_games, shrink_k=shrink_k, carry_fraction=carry_fraction)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ratings_df.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(ratings_df), OUTPUT_PATH)

    iters = pd.DataFrame(iteration_log)
    print("\nOpponent-adjustment iterations used (by split):")
    print(iters.groupby("split")["iterations"].agg(["min", "mean", "max"]).to_string())
    print(f"Boundaries hitting the {MAX_ITER}-iteration cap: {int((iters.iterations == MAX_ITER).sum())} / {len(iters)}")

    print("\nRating distributions by season (overall_off_epa / overall_def_epa):")
    for season, g in ratings_df.groupby("season"):
        off, deff = g["overall_off_epa"], g["overall_def_epa"]
        print(f"  {season}: off mean={off.mean():+.4f} sd={off.std():.4f} min={off.min():+.4f} max={off.max():+.4f}  |  "
              f"def mean={deff.mean():+.4f} sd={deff.std():.4f} min={deff.min():+.4f} max={deff.max():+.4f}")

    sanity_season = 2024
    final_week = ratings_df.loc[ratings_df.season == sanity_season, "week"].max()
    snap = ratings_df[(ratings_df.season == sanity_season) & (ratings_df.week == final_week)].copy()
    snap["net_epa"] = snap.overall_off_epa - snap.overall_def_epa
    snap = snap.sort_values("net_epa", ascending=False)
    print(f"\nSanity check — season {sanity_season}, week {final_week} (last), overall net rating (off - def):")
    print("  Top 3:")
    for row in snap.head(3).itertuples():
        print(f"    {row.team}: off={row.overall_off_epa:+.4f} def={row.overall_def_epa:+.4f} net={row.net_epa:+.4f}")
    print("  Bottom 3:")
    for row in snap.tail(3).itertuples():
        print(f"    {row.team}: off={row.overall_off_epa:+.4f} def={row.overall_def_epa:+.4f} net={row.net_epa:+.4f}")

    return ratings_df


if __name__ == "__main__":
    main()
