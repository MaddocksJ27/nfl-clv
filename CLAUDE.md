# NFL CLV & Anytime TD Model

## Goal
Two spread targets:
1. BASELINE — predict closing spread from team strength (opponent-adjusted EPA).
   Expect high R². Its real job is producing implied team points for the TD layer.
2. HEADLINE — predict open-to-close drift (close − open). Small signal, ~10-20%
   variance explained is a good result. This is the tradeable model.
Then: anytime-TD scorer probabilities from implied team points.

Primary metric is CLV, not ATS win rate. Portfolio project — every choice must be
defensible in a technical interview.

## Data
- nflverse via nfl_data_py: play-by-play, schedules, injuries, snap counts
- The Odds API: spreads/totals history (2020+), player props (2023+), UK region
- Openers: sportsbookreviewsonline Donbest archive

## Rules
- No look-ahead. Every feature computed as-of the prediction timestamp.
- Walk-forward validation only. Never random k-fold.
- Notebooks explore; anything that runs lives in src/.
- Secrets in .env, never committed.
- Ask before adding dependencies.

## Stack
Python 3.13, pandas, LightGBM, statsmodels. macOS, Europe/London timezone.
