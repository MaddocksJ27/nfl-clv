# NFL Betting Market Efficiency

An empirical study of where NFL betting markets are efficient and where they are not, using six seasons of line history and three seasons of player prop prices.

The project started as an attempt to build a profitable model. It became a study of *why that is hard*, with each hypothesis tested and reported regardless of outcome. Every result below is out-of-sample.

---

## Summary of findings

| Question | Result |
|---|---|
| Can team strength predict the closing spread? | Yes — R² = 0.634, RMSE 3.5 pts, walk-forward |
| Can 3-day line drift be forecast? | No — R² ≈ 0 across 804 out-of-sample games |
| Are anytime-TD prices well calibrated? | Yes — predicted matches actual in every decile, every position |
| Are "soft" books soft enough to beat? | No — they lose to their own vig |
| Do books misprice skewed prop distributions? | No — they price the median, not the mean |
| Is the apparent Under bias exploitable? | No — it is line-shopping value, not mispricing |
| Do books under-adjust rush attempts for game script? | Slightly — ~11% shallower slope than reality, too thin to bet |

---

## Data

| Source | Coverage | Notes |
|---|---|---|
| nflverse (`nfl_data_py`) | 2016–2025 | Play-by-play, schedules, injuries, snap counts, depth charts, rosters |
| The Odds API — featured markets | 2020–2025, daily snapshots | 783 snapshots, 633,868 rows, 1,734 events |
| The Odds API — player props | 2023–2025, closing snapshot | 855 games, 726,738 rows, 6 markets, 7 books |
| SBR archive (scraped) | 2016–2021 | Opening/closing spreads; archive is frozen and no longer updated |

Total spend on odds data: ~$59 for one month of API access.

---

## Methodology

The project's constraints, applied throughout:

**No look-ahead.** Every feature is computed strictly from information available at the prediction timestamp. This is enforced by tests, not by convention — see `tests/test_epa_no_lookahead.py`, which asserts that ratings are byte-identical when recomputed from only strictly-earlier plays, that week-1 ratings depend on zero plays from that season, and that shuffling future games leaves historical ratings unchanged.

**Walk-forward validation only.** Train on seasons strictly before the test season. No random splits anywhere in the repo.

**Stratified reporting.** Every quality metric is reported by season, and by market or position where relevant. This convention caught three separate bugs that were invisible in aggregate — see *Bugs found* below.

**Robustness before belief.** Promising results were subjected to tests designed to kill them: season splits, bootstrap lower bounds, deduplication, and mirror checks on the opposite side of the bet. Several did not survive.

---

## 1. Predicting the closing spread

**Target:** consensus closing spread, home perspective.
**Features:** opponent-adjusted EPA ratings (offense/defense), projected-starter QB rating, QB change flag, injury status, rest days, divisional flag, roof, week.

Ratings are built by exponentially-weighted, opponent-adjusted iteration over prior games, shrunk toward league mean, with a decaying carry-in from the prior season.

**Walk-forward results (OLS):**

| Season | RMSE | MAE | R² |
|---|---|---|---|
| 2021 | 3.98 | 2.94 | 0.636 |
| 2022 | 3.90 | 2.99 | 0.538 |
| 2023 | 3.66 | 2.91 | 0.582 |
| 2024 | 3.67 | 2.75 | 0.542 |
| **Pooled** | **3.53** | **2.72** | **0.634** |

LightGBM underperformed OLS at every fold (pooled R² 0.603). With 250–1,100 training rows per fold this is a statement about sample size, not about the underlying relationship being linear.

### Notes on this model

**QB identity is the single largest feature.** Adding projected-starter QB features moved R² from 0.580 to 0.634. Coefficients are sign-correct and symmetric: home QB rating −9.11, away +8.99.

**Starter identification must respect information timing.** An earlier version identified starters from snap counts, which only exist *after* the game. Replacing this with a pre-kickoff projection (previous starter → depth chart → injury-report override) cost only 0.003 R², but the original version was using information unavailable at bet time. The projected starter differs from the actual starter in 10.7% of team-games.

**Filtering to "clean" plays made the model worse.** Restricting EPA to early downs, neutral win probability, and excluding end-of-half situations dropped R² from 0.637 to 0.476. This is a real result: the target is *the market's price*, and the market prices teams on their full body of work. Filtering toward true talent moves away from what books price, not toward it.

**No detectable 2020 home-field effect.** A no-crowd interaction term was insignificant (coefficient −0.061, p = 0.805).

---

## 2. Is line movement forecastable?

**Target:** `close − consensus spread at kickoff minus 3 days`.

### Defining the target correctly

The naive definition — last observed price minus first observed price — is contaminated. Books post lines at wildly varying horizons: median 10 days before kickoff, mean 30.6, with 36% of games first quoted more than three weeks out. Under that definition, drift SD was 3.22 points and appeared to rise from 1.77 (2021) to 4.30 (2024).

That trend was an artefact. A line posted eight weeks out moves mainly because *the teams play games in between* — that is team-strength revision, not pre-kickoff information flow.

Re-anchoring to a fixed 3-day horizon gives a stable quantity:

| Horizon | n | Drift SD | Season range |
|---|---|---|---|
| First observed | 1,391 | 3.22 | 1.77 – 4.30 |
| Kickoff − 7d | 917 | 2.64 | 2.15 – 3.80 |
| **Kickoff − 3d** | **1,380** | **0.955** | **0.81 – 1.16** |

The 3-day SD is stable across all five seasons, confirming it measures one quantity rather than a mixture. The tightening from 1.16 (2020) to 0.81 (2023) is visible evidence of the market sharpening post-legalisation.

### Result

Ridge regression, alpha selected within training folds only, four features including the divergence between the fundamental model's prediction and the market price:

| Feature set | n | RMSE | MAE | R² |
|---|---|---|---|---|
| Market state (4 features) | 804 | 0.846 | 0.551 | −0.002 |
| Prior movement (3 features) | 804 | 0.853 | 0.550 | −0.020 |
| Combined (7 features) | 804 | 0.853 | 0.557 | −0.019 |

**Three-day spread movement is not forecastable from public fundamentals.**

Notably, `model_prediction − market_price` carries no information about subsequent movement. The market diverging from a fundamental estimate does not mean it is about to converge to it.

### Tail analysis

Restricting to |drift| > 1.5 (n = 84, 8% of games):

- **Momentum:** the sign of the prior 2-day move matches the subsequent move 63.7% of the time (n = 80). The 95% CI is 52.7%–74.7%, and one season (2023, n = 19, 84%) carries the pooled figure. The subgroup is also defined by the outcome being predicted, so this is not directly tradeable.
- **Book disagreement:** cross-book price dispersion at the horizon is *negatively* correlated with subsequent drift magnitude (Pearson −0.25, Spearman −0.29, monotonic across quartiles). The opposite of the naive hypothesis. Interpretation: disagreement indicates the move has already partly happened, not that one is coming.

---

## 3. Anytime touchdown prices

### De-vigging a non-exclusive field

Anytime TD is not a mutually exclusive market — a mean of 4.11 distinct skill-position players score per game. Normalising implied probabilities to sum to 1.0 (standard for win markets) is the wrong operation and produces a systematic 4× scale error. Shin's method is likewise inapplicable, as it assumes an exclusive field.

The correct treatment estimates a per-leg multiplicative margin:

```
m = (sum of raw implied probabilities) / E[distinct scorers]
```

with `E[distinct scorers]` fit walk-forward from the game total. Estimated margin: **m ≈ 1.18**.

### Calibration

After correct de-vigging, market prices are well calibrated across all deciles and all four positions:

| Decile | Predicted | Actual |
|---|---|---|
| 1 | 0.034 | 0.027 |
| 5 | 0.126 | 0.122 |
| 10 | 0.463 | 0.477 |

There is no favourite-longshot bias to exploit. Scratches were excluded using snap-count participation (9.5% of priced rows); an earlier play-by-play-based definition wrongly flagged 22.8% by classifying blocking fullbacks and zero-touch tight ends as inactive.

### Cross-book disagreement

The peer median outperforms outlier books at every threshold (Brier 0.162 vs 0.164 at 2pp, widening to 0.172 vs 0.191 at 6pp). BetRivers and BetMGM are outliers most often and are consistently softer; FanDuel and DraftKings diverge least and are the only books ever sharper than the median.

**But relative sharpness is not absolute edge.** Backing every below-median outlier at that book's actual price:

| Threshold | n | Hit rate | Implied p | ROI |
|---|---|---|---|---|
| 2pp | 7,376 | 27.5% | 29.5% | −6.30% |
| 4pp | 1,335 | 33.9% | 36.5% | −4.39% |
| 6pp | 308 | 37.3% | 39.5% | −5.09% |

The soft books are softer than their peers but not softer than their own margin.

Additionally, **the market is one-sided**: only 0.9% of anytime-TD rows carry a "No" price, and zero of 9,515 above-median outlier cases had one. Overpriced players cannot be laid. This closes the market structurally, not just empirically.

---

## 4. Volume props: how books set prop lines

Five Over/Under markets: receptions, rush attempts, rushing yards, receiving yards, passing yards.

### Line disagreement varies enormously by market type

| Market | Games with line disagreement | Median gap |
|---|---|---|
| Receptions | 20% | 1 |
| Rush attempts | 42% | 1 |
| Rushing yards | 89% | 2 yds |
| Receiving yards | 91% | 2 yds |
| Passing yards | 99.7% | 6 yds |

Count markets are near-consensus; continuous yardage markets are not.

### Books price the median, not the mean

Four of five markets are right-skewed (receiving yards skew +1.53, rushing yards +1.42, receptions +1.04). Passing yards is nearly symmetric (+0.03) — a sum over many plays, so approximately normal.

If books set lines at the conditional *mean*, skew alone would push Under hit rates to 57–64%. Observed Under rates are 52–54%.

| Market | P(Under) if line = mean | Observed P(Under) |
|---|---|---|
| Receptions | 0.641 | 0.531 |
| Rush attempts | 0.574 | 0.531 |
| Rushing yards | 0.607 | 0.536 |
| Receiving yards | 0.603 | 0.523 |
| Passing yards | 0.498 | 0.497 |

Consensus lines track the conditional median closely and sit well below the conditional mean. **Books have already absorbed the skew.** The near-symmetric passing-yards market acts as a natural control, showing no gap — exactly as the explanation predicts.

### The Under "edge" is line shopping, not mispricing

Across the full population (546,816 quotes), both sides lose to vig, but asymmetrically:

| Side | n | Mean implied p | Hit rate | Shortfall |
|---|---|---|---|---|
| Under | 272,720 | 0.550 | 0.531 | 1.9pp |
| Over | 274,096 | 0.519 | 0.469 | 5.0pp |

Selecting below-median outlier prices produced apparently positive Under ROI (+0.3%, +3.8%, +8.4% at 2/4/6pp) with a clean dose-response and a mirror-image Over result. It did not survive scrutiny:

- **Season split:** 2023 negative at 2pp
- **Market split:** driven by rush attempts and rushing yards; passing yards strongly negative
- **Bootstrap:** 5th percentile below zero at all three thresholds
- **Deduplication:** counting each book's quote separately inflated n roughly 7×. At one consensus price per player-game, no bucket in any market has a bootstrap lower bound above zero.

**Conclusion: the apparent edge was value from finding the softest individual quote, not from predicting outcomes.** That distinction — execution advantage versus predictive skill — is the central methodological point of this project.

### Game script and rush attempts

Books do under-adjust rush-attempt lines for game script, but barely: the line's slope against spread is +0.098 attempts per point versus +0.110 in reality — about 11% shallower. Both regressions have R² under 0.02. Real, statistically detectable, and far too thin to bet.

---

## Bugs found

Documented because finding them is the point, and each was caught by checking rather than trusting an aggregate number.

1. **Postseason roster lookup.** Roster pools were built with `game_type="REG"`, but nflverse labels playoff weeks `WC`/`DIV`/`CON`/`SB`. All 38 postseason games had 100% unresolvable player names while the overall match rate looked healthy. Caught by stratified reporting.

2. **Odds API event-ID reissue.** The API assigns a provisional event ID to far-future games and a new one closer to kickoff. 265 of 1,673 events were affected, concentrated in 2023, inflating that season's game count to 474 versus ~285 elsewhere. Fixed by consolidating on nflverse `game_id`.

3. **Oscillating opponent adjustment.** Simultaneous (Jacobi) offense/defense updates oscillate indefinitely rather than converging. Fixed with Gauss-Seidel ordering plus damping.

4. **Invalid convergence criterion.** Under damping, step sizes shrink geometrically whether or not the iteration has converged, so a small final-step delta does not imply proximity to the fixed point. Early-season boundaries flagged as converged were up to 0.60 EPA/play from their true value — three times the typical team spread.

5. **Length-biased fuzzy name matching.** `difflib` ratio favoured a short wrong name over a long right one: `M. Jones Jr.` matched Cam Jones (LB) over Marvin Jones (WR). Fixed with a dedicated initial-plus-surname tier ahead of the generic fallback. Regression test added.

6. **Fullback position labels.** Snap counts label fullbacks `FB`, not `RB`. A position filter caused every starting fullback to be classified as a scratch on every appearance.

---

## What this project does not claim

- **No profitable strategy was found.** Every candidate edge failed at least one robustness test.
- **Calibration was measured against US closing prices.** UK books returned no historical prop data, so a model calibrated here is unproven against the books a UK bettor would actually use.
- **No sharp reference price.** The Odds API does not carry Pinnacle or exchange prices, so "the closing line" here means a soft-book consensus, which is a weaker benchmark than the sharpest available price.
- **Props were captured only at the close.** The market's sharpest moment. Whether earlier prices are softer is untested — see below.

---

## Open question

Every efficiency result here is measured against *closing* prices. Prop markets are posted several days before kickoff, at lower limits and with less attention, and the question of whether early prices are meaningfully softer is untested by this study. That is the next experiment.

---

## Repository

```
src/
  ingest/     nflverse, Odds API client, bulk pulls, team name normalisation
  features/   EPA ratings, player name resolution
  models/     spread baseline, drift
  eval/       validation, calibration, cross-book disagreement
tests/        no-lookahead assertions, resolver regressions
```

Requires an Odds API key in `.env` as `ODDS_API_KEY`. Raw and interim data are gitignored; ingestion modules cache to parquet and resume from cache.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.ingest.nflverse
```
