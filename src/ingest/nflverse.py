"""Fetch nflverse datasets and cache them locally as parquet.

Fetch and store only — no cleaning or feature engineering here.
"""

import logging
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEASONS = list(range(2016, 2026))

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "spread_line",
    "total_line",
    "home_moneyline",
    "away_moneyline",
    "roof",
    "surface",
    "home_rest",
    "away_rest",
    "div_game",
]


def _load_or_fetch(path: Path, fetch_fn, force_refresh: bool = False) -> pd.DataFrame:
    if path.exists() and not force_refresh:
        logger.info("Reading cached %s", path)
        return pd.read_parquet(path)

    df = fetch_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Cached %s (%d rows)", path, len(df))
    return df


def _log_coverage(name: str, df: pd.DataFrame, season_col: str = "season") -> None:
    if season_col in df.columns:
        seasons = sorted(df[season_col].dropna().unique().tolist())
        logger.info("%s: %d rows, seasons %s-%s", name, len(df), seasons[0], seasons[-1])
    else:
        logger.info("%s: %d rows, no season column", name, len(df))


def fetch_schedules(force_refresh: bool = False) -> pd.DataFrame:
    path = RAW_DIR / "schedules.parquet"

    def _fetch():
        df = nfl.import_schedules(SEASONS)
        return df[SCHEDULE_COLUMNS]

    df = _load_or_fetch(path, _fetch, force_refresh)
    _log_coverage("schedules", df)
    return df


def fetch_pbp(force_refresh: bool = False) -> pd.DataFrame:
    path = RAW_DIR / "pbp.parquet"

    def _fetch():
        return nfl.import_pbp_data(SEASONS, downcast=True)

    df = _load_or_fetch(path, _fetch, force_refresh)
    _log_coverage("pbp", df)
    return df


def fetch_injuries(force_refresh: bool = False) -> pd.DataFrame:
    path = RAW_DIR / "injuries.parquet"

    def _fetch():
        return nfl.import_injuries(SEASONS)

    df = _load_or_fetch(path, _fetch, force_refresh)
    _log_coverage("injuries", df)
    return df


def fetch_snap_counts(force_refresh: bool = False) -> pd.DataFrame:
    path = RAW_DIR / "snap_counts.parquet"

    def _fetch():
        return nfl.import_snap_counts(SEASONS)

    df = _load_or_fetch(path, _fetch, force_refresh)
    _log_coverage("snap_counts", df, season_col="season")
    return df


def fetch_rosters(force_refresh: bool = False) -> pd.DataFrame:
    path = RAW_DIR / "rosters.parquet"

    def _fetch():
        # nfl_data_py 0.3.2's import_weekly_rosters raises "cannot reindex on
        # an axis with duplicate labels" when given more than one season at
        # once (an age-calculation bug in that library version) — fetch one
        # season at a time and concatenate as a workaround.
        return pd.concat(
            [nfl.import_weekly_rosters([season]) for season in SEASONS],
            ignore_index=True,
        )

    df = _load_or_fetch(path, _fetch, force_refresh)
    _log_coverage("rosters", df, season_col="season")
    return df


def main(force_refresh: bool = False) -> None:
    fetch_schedules(force_refresh)
    fetch_pbp(force_refresh)
    fetch_injuries(force_refresh)
    fetch_snap_counts(force_refresh)
    fetch_rosters(force_refresh)


if __name__ == "__main__":
    main()
