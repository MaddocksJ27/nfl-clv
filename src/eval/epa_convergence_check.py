"""Diagnostic follow-up on the 93 (season, week, split) boundaries that hit
the 20-iteration cap in the epa_ratings opponent-adjustment solve.

For each capped boundary:
1. Report the final iteration's max single-team change (how far from the
   1e-3 convergence tolerance it actually was when it got cut off).
2. Recompute with a 200-iteration cap and report the max difference vs the
   20-iteration values, across all teams and both off/def — i.e. how much
   the rating would have moved had it been allowed to keep going.
"""

import pandas as pd

from src.features.epa_ratings import (
    load_plays, build_team_games, _mu_asof, _opponent_adjust,
    DEFAULT_HALF_LIFE_GAMES, MAX_ITER,
)

EXTENDED_MAX_ITER = 200


def find_capped_boundaries(half_life_games: float = DEFAULT_HALF_LIFE_GAMES):
    """Re-derive the same (season, week, split) list and per-team real-game
    slices used by the main pipeline's reported iteration_log — the output
    weeks only (not the internal "final rating" pseudo-boundary used for
    season-to-season carry-in, which the pipeline doesn't log iterations
    for), so this reproduces exactly the same 645/93 counts reported
    earlier. The opponent-adjustment solve only touches within-season real
    games, independent of the carry-in, so no season-chain bookkeeping is
    needed here regardless."""
    plays = load_plays()
    team_games = build_team_games(plays)

    capped = []
    for season in sorted(team_games.season.unique()):
        weeks = sorted(team_games.loc[team_games.season == season, "week"].unique())
        for week in weeks:
            real_games = team_games[(team_games.season == season) & (team_games.week < week)]
            for split in ("overall", "pass", "run"):
                mu = _mu_asof(team_games, split, season, week)
                *_, iters, final_max_change = _opponent_adjust(real_games, split, mu, half_life_games, max_iter=MAX_ITER)
                if iters == MAX_ITER:
                    capped.append({
                        "season": season, "week": week, "split": split, "mu": mu,
                        "final_max_change_20iter": final_max_change,
                    })
    return pd.DataFrame(capped), team_games


def main():
    capped_df, team_games = find_capped_boundaries()
    print(f"Boundaries hitting the {MAX_ITER}-iteration cap: {len(capped_df)}")
    print()
    print("Final-iteration max single-team change (20-iteration cap):")
    print(capped_df["final_max_change_20iter"].describe().to_string())
    print(f"\nOverall max across all {len(capped_df)} capped boundaries: "
          f"{capped_df['final_max_change_20iter'].max():.6f}")

    diffs = []
    for row in capped_df.itertuples():
        real_games = team_games[(team_games.season == row.season) & (team_games.week < row.week)]

        off_20, def_20, *_ = _opponent_adjust(real_games, row.split, row.mu, DEFAULT_HALF_LIFE_GAMES, max_iter=MAX_ITER)
        off_200, def_200, *_ = _opponent_adjust(real_games, row.split, row.mu, DEFAULT_HALF_LIFE_GAMES, max_iter=EXTENDED_MAX_ITER)

        teams = set(off_20) | set(off_200)
        max_diff = 0.0
        for t in teams:
            max_diff = max(max_diff, abs(off_20.get(t, 0.0) - off_200.get(t, 0.0)))
            max_diff = max(max_diff, abs(def_20.get(t, 0.0) - def_200.get(t, 0.0)))

        diffs.append({"season": row.season, "week": row.week, "split": row.split, "max_diff_20_vs_200": max_diff})

    diffs_df = pd.DataFrame(diffs)
    print(f"\nMax rating difference, {MAX_ITER}-iteration vs {EXTENDED_MAX_ITER}-iteration cap:")
    print(diffs_df["max_diff_20_vs_200"].describe().to_string())
    worst = diffs_df.sort_values("max_diff_20_vs_200", ascending=False).head(5)
    print("\nWorst 5 boundaries:")
    print(worst.to_string(index=False))
    print(f"\nOverall max difference across all {len(diffs_df)} capped boundaries: "
          f"{diffs_df['max_diff_20_vs_200'].max():.6f}")


if __name__ == "__main__":
    main()
