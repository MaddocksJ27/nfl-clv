"""No-lookahead guarantees for src.features.epa_ratings.

Uses stdlib unittest (no pytest in requirements.txt). Reads the cached
data/raw/pbp.parquet and data/interim/epa_ratings.parquet — run
`python -m src.features.epa_ratings` first if the latter doesn't exist.
"""

import random
import unittest
from pathlib import Path

import pandas as pd

from src.features.epa_ratings import (
    ALL_TEAMS, SPLITS, compute_ratings_asof, load_plays, DEFAULT_HALF_LIFE_GAMES,
    DEFAULT_SHRINK_K, DEFAULT_CARRY_FRACTION,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RATINGS_PATH = REPO_ROOT / "data" / "interim" / "epa_ratings.parquet"

TOLERANCE = 1e-9
RANDOM_SEED = 42
N_SAMPLES = 50


def _rating_columns():
    cols = []
    for split in SPLITS:
        cols.append(f"{split}_off_epa")
        cols.append(f"{split}_def_epa")
    return cols


class TestReproducibleFromRestrictedPlays(unittest.TestCase):
    """50 random (season, week, team) rows: recompute from scratch, passed
    ONLY plays strictly earlier than (season, week), and confirm it matches
    the value in the precomputed ratings parquet exactly."""

    @classmethod
    def setUpClass(cls):
        if not RATINGS_PATH.exists():
            raise unittest.SkipTest(f"{RATINGS_PATH} not found — run the pipeline first")
        cls.ratings = pd.read_parquet(RATINGS_PATH)
        cls.all_plays = load_plays()

        random.seed(RANDOM_SEED)
        idx = random.sample(range(len(cls.ratings)), min(N_SAMPLES, len(cls.ratings)))
        cls.sample_rows = cls.ratings.iloc[idx].to_dict("records")

    def test_fifty_random_rows_reproduce_from_strictly_earlier_plays(self):
        mismatches = []
        for row in self.sample_rows:
            season, week, team = row["season"], row["week"], row["team"]

            restricted = self.all_plays[
                (self.all_plays.season < season) | ((self.all_plays.season == season) & (self.all_plays.week < week))
            ]
            recomputed = compute_ratings_asof(
                restricted, season, week,
                half_life_games=DEFAULT_HALF_LIFE_GAMES, shrink_k=DEFAULT_SHRINK_K,
                carry_fraction=DEFAULT_CARRY_FRACTION, already_filtered=True,
            )

            for col in _rating_columns():
                # col looks like "overall_off_epa" -> split="overall", side="off"
                split = col.replace("_off_epa", "").replace("_def_epa", "")
                side = "off" if col.endswith("_off_epa") else "def"
                expected = row[col]
                actual = recomputed[team][split][side]
                if abs(expected - actual) > TOLERANCE:
                    mismatches.append(
                        f"season={season} week={week} team={team} {col}: "
                        f"stored={expected!r} recomputed={actual!r} diff={abs(expected - actual):.2e}"
                    )

        self.assertEqual(
            mismatches, [],
            "Ratings not reproducible from strictly-earlier plays alone:\n" + "\n".join(mismatches),
        )


class TestWeekOneDependsOnZeroCurrentSeasonPlays(unittest.TestCase):
    """Every season's week-1 rating must be identical whether or not that
    season's plays are present in the input at all — removing them
    entirely (not just filtering by week) can't change the answer if the
    algorithm truly never touches them."""

    @classmethod
    def setUpClass(cls):
        cls.all_plays = load_plays()
        cls.seasons = sorted(cls.all_plays.season.unique().tolist())

    def test_week1_unaffected_by_removing_that_seasons_plays_entirely(self):
        random.seed(RANDOM_SEED)
        sample_seasons = random.sample(self.seasons[1:], min(3, len(self.seasons) - 1))  # skip the very first season

        for season in sample_seasons:
            with_season = compute_ratings_asof(self.all_plays, season, 1, already_filtered=True)
            without_season = compute_ratings_asof(
                self.all_plays[self.all_plays.season != season], season, 1, already_filtered=True,
            )

            mismatches = []
            for team in ALL_TEAMS:
                for split in SPLITS:
                    for side in ("off", "def"):
                        a = with_season[team][split][side]
                        b = without_season[team][split][side]
                        if abs(a - b) > TOLERANCE:
                            mismatches.append(f"season={season} team={team} {split}_{side}: {a!r} vs {b!r}")

            self.assertEqual(
                mismatches, [],
                f"Week 1 of season {season} changed when that season's plays were removed entirely:\n"
                + "\n".join(mismatches),
            )


class TestFutureGameOrderDoesNotAffectHistoricalRatings(unittest.TestCase):
    """Shuffling the row order of games at/after a target week must not
    change the rating computed as-of that week."""

    def test_shuffled_future_rows_leave_historical_rating_unchanged(self):
        all_plays = load_plays()
        seasons = sorted(all_plays.season.unique().tolist())
        target_season = seasons[len(seasons) // 2]
        target_week = 10

        baseline = compute_ratings_asof(all_plays, target_season, target_week, already_filtered=True)

        is_future = (all_plays.season > target_season) | (
            (all_plays.season == target_season) & (all_plays.week >= target_week)
        )
        past = all_plays[~is_future]
        future_shuffled = all_plays[is_future].sample(frac=1, random_state=RANDOM_SEED)
        shuffled_plays = pd.concat([past, future_shuffled], ignore_index=True)

        shuffled_result = compute_ratings_asof(shuffled_plays, target_season, target_week, already_filtered=True)

        mismatches = []
        for team in ALL_TEAMS:
            for split in SPLITS:
                for side in ("off", "def"):
                    a = baseline[team][split][side]
                    b = shuffled_result[team][split][side]
                    if abs(a - b) > TOLERANCE:
                        mismatches.append(f"team={team} {split}_{side}: {a!r} vs {b!r}")

        self.assertEqual(
            mismatches, [],
            f"Shuffling future game order changed the season={target_season} week={target_week} rating:\n"
            + "\n".join(mismatches),
        )


if __name__ == "__main__":
    unittest.main()
