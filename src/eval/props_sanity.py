"""Sanity-check the anytime-TD props market: overround, de-vig, and
calibration against actual touchdown scorers.

Pure offline analysis — no network calls, only cached parquet files.

## Why "sum across all players" instead of per-player Yes/No de-vig

The obvious way to de-vig a 2-sided market is per-outcome (Yes price vs No
price). That doesn't work here: only 1,479/169,345 (0.9%) of anytime-TD rows
have a "No" side priced at all — books only quote "Yes" for this market.

So de-vigging has to work the way it's actually done for these "anytime
scorer" markets in practice (this is the standard treatment in the market-
efficiency literature the task references, e.g. github.com/mberk/shin, MIT
licensed, whose calculate_implied_probabilities() takes a full list of odds
for one market and returns de-vigged probabilities + z): treat every
player's Yes-price in a (game_id, book) as one N-way field and de-vig them
together, exactly as if this were a mutually-exclusive market (only one
runner wins). It isn't really mutually exclusive — multiple players DO
score in the same game — so both de-vig methods here will, by construction,
force each (game_id, book)'s de-vigged probabilities to sum to ~1.0. That's
not a bug: comparing that trivial ~1.0 against the real, much larger number
of distinct actual scorers per game (part 3/4) is the point — it exposes
that mutual-exclusivity de-vigging is the wrong model for this market type,
and should show up again in part 5 as systematic under-prediction in the
calibration table.

## Shin's method (implemented from the standard formula, not literally
vendored from mberk/shin, to keep this self-contained and dependency-free)

For raw implied probabilities p_i = 1/decimal_odds_i over n outcomes with
R = sum(p_i), Shin's model solves for z in:
    sum_i [ sqrt(z^2 + 4(1-z) p_i^2 / R) - z ] / (2(1-z)) = 1
and the de-vigged probability for outcome i is that same per-i term. Verified
by hand for the symmetric 2-outcome case (p1=p2=0.55, R=1.10): solving gives
z=0.1, matching the well-known z ~= R-1 approximation for small, symmetric
2-way overrounds.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

REPO_ROOT = Path(__file__).resolve().parents[2]
PROPS_PATH = REPO_ROOT / "data" / "interim" / "props.parquet"
EVENT_INDEX_PATH = REPO_ROOT / "data" / "raw" / "props" / "_event_index.parquet"
PBP_PATH = REPO_ROOT / "data" / "raw" / "pbp.parquet"
ROSTERS_PATH = REPO_ROOT / "data" / "raw" / "rosters.parquet"

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
ACTUAL_SCORER_SEASONS = (2023, 2024, 2025)

# Columns used to decide "did this player appear in this game's pbp at all"
# (scratch detection) — every offensive way a skill player leaves a trace.
PARTICIPATION_COLUMNS = [
    "passer_player_id", "rusher_player_id", "receiver_player_id", "td_player_id",
    "fumbled_1_player_id", "fumbled_2_player_id",
    "lateral_receiver_player_id", "lateral_rusher_player_id",
]


def load_td_props() -> pd.DataFrame:
    """anytime-TD, Yes side, QB/RB/WR/TE only, with game_id/season/week
    joined in from the event index."""
    props = pd.read_parquet(PROPS_PATH)
    td = props[
        (props.market == "player_anytime_td")
        & (props.side == "Yes")
        & props.position.isin(SKILL_POSITIONS)
        & props.player_id.notna()
    ].copy()

    idx = pd.read_parquet(EVENT_INDEX_PATH)[["event_id", "game_id", "season", "week"]]
    td = td.merge(idx, on="event_id", how="inner")
    td["p_raw"] = 1.0 / td["price"]
    return td


def shin_devig(p_raw: np.ndarray) -> tuple:
    """p_raw: raw implied probabilities for one (game_id, book) field.
    Returns (devigged_probs, z). Falls back to proportional (z=0 model,
    equivalent when there's no overround) if R<=1 or the field has a
    single outcome (z undefined)."""
    R = p_raw.sum()
    if R <= 1.0 or len(p_raw) < 2:
        return p_raw / R, 0.0

    def f(z):
        return np.sum((np.sqrt(z ** 2 + 4 * (1 - z) * p_raw ** 2 / R) - z) / (2 * (1 - z))) - 1.0

    # f(0) = sqrt(R) - 1 > 0 for R>1; f approaches a finite limit < 0 as z->1^-,
    # so a root exists in (0, 1) whenever there's a real overround to remove.
    try:
        z = brentq(f, 1e-9, 1 - 1e-9)
    except ValueError:
        return p_raw / R, 0.0
    devigged = (np.sqrt(z ** 2 + 4 * (1 - z) * p_raw ** 2 / R) - z) / (2 * (1 - z))
    return devigged, z


def compute_devig(td: pd.DataFrame) -> pd.DataFrame:
    """Adds p_prop (proportional de-vig) and p_shin (Shin de-vig) columns,
    plus a `shin_z` column (repeated per row within a (game_id, book) group)."""
    td = td.sort_values(["game_id", "book"]).reset_index(drop=True)
    p_prop = np.empty(len(td))
    p_shin = np.empty(len(td))
    z_col = np.empty(len(td))

    for _, idx in td.groupby(["game_id", "book"]).groups.items():
        pos = td.index.get_indexer(idx)
        raw = td.loc[idx, "p_raw"].to_numpy()
        R = raw.sum()
        p_prop[pos] = raw / R
        devigged, z = shin_devig(raw)
        p_shin[pos] = devigged
        z_col[pos] = z

    td["p_prop"] = p_prop
    td["p_shin"] = p_shin
    td["shin_z"] = z_col
    return td


def compute_actual_scorers(seasons=ACTUAL_SCORER_SEASONS) -> pd.DataFrame:
    """One row per (game_id, player_id) that actually scored a rushing or
    receiving touchdown, restricted to QB/RB/WR/TE and the given seasons."""
    cols = ["season", "game_id", "touchdown", "td_player_id"]
    pbp = pd.read_parquet(PBP_PATH, columns=cols)
    pbp = pbp[pbp.season.isin(seasons) & (pbp.touchdown == 1) & pbp.td_player_id.notna()]

    rosters = pd.read_parquet(ROSTERS_PATH, columns=["season", "player_id", "position"])
    pos_lookup = rosters.drop_duplicates(subset=["season", "player_id"]).set_index(["season", "player_id"])["position"]

    scorers = pbp[["season", "game_id", "td_player_id"]].drop_duplicates()
    scorers["position"] = scorers.set_index(["season", "td_player_id"]).index.map(pos_lookup)
    scorers = scorers[scorers.position.isin(SKILL_POSITIONS)]
    return scorers.rename(columns={"td_player_id": "player_id"})[["season", "game_id", "player_id", "position"]]


def compute_participation_sets(seasons=ACTUAL_SCORER_SEASONS) -> dict:
    """{game_id: set(player_id)} — every player_id appearing in any
    offensive participation column, for scratch detection."""
    cols = ["season", "game_id"] + PARTICIPATION_COLUMNS
    pbp = pd.read_parquet(PBP_PATH, columns=cols)
    pbp = pbp[pbp.season.isin(seasons)]

    sets = {}
    for col in PARTICIPATION_COLUMNS:
        sub = pbp[["game_id", col]].dropna(subset=[col])
        for game_id, group in sub.groupby("game_id")[col]:
            sets.setdefault(game_id, set()).update(group.unique())
    return sets


def report_overround(td: pd.DataFrame) -> None:
    print("=" * 70)
    print("PART 1 — Raw overround: sum of raw p(Yes) across all players, per (game, book)")
    print("=" * 70)
    per_game_book = td.groupby(["game_id", "book"])["p_raw"].sum()
    print(f"n = {len(per_game_book)} (game, book) pairs")
    print(per_game_book.describe(percentiles=[0.25, 0.5, 0.75]).to_string())


def report_devig(td: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PART 2 — De-vigged summed probability, per (game, book)")
    print("=" * 70)

    for col, label in [("p_prop", "Proportional"), ("p_shin", "Shin")]:
        per_game_book = td.groupby(["game_id", "book"])[col].sum()
        print(f"\n{label} de-vig — summed probability distribution:")
        print(per_game_book.describe(percentiles=[0.25, 0.5, 0.75]).to_string())

    z_per_market = td.groupby(["game_id", "book"])["shin_z"].first()
    print("\nShin's z — distribution across (game, book) markets:")
    print(z_per_market.describe(percentiles=[0.25, 0.5, 0.75]).to_string())


def report_actual_scorers(scorers: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(f"PART 3 — Actual distinct QB/RB/WR/TE TD scorers per game, {ACTUAL_SCORER_SEASONS}")
    print("=" * 70)
    per_game = scorers.groupby("game_id")["player_id"].nunique()
    print(f"n = {len(per_game)} games")
    print(f"mean = {per_game.mean():.3f}")
    print("\nFull distribution:")
    print(per_game.value_counts().sort_index().to_string())
    return per_game.rename("actual_scorers").reset_index()


def report_comparison(td: pd.DataFrame, per_game_actual: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PART 4 — De-vigged sum (part 2) vs actual distinct scorers (part 3)")
    print("=" * 70)

    devig_sum = td.groupby(["game_id", "book", "season"])[["p_prop", "p_shin"]].sum().reset_index()
    merged = devig_sum.merge(per_game_actual, on="game_id", how="inner")

    print("\nBy season (mean across books):")
    by_season = merged.groupby("season")[["p_prop", "p_shin", "actual_scorers"]].mean()
    print(by_season.to_string())

    print("\nBy book (mean across seasons):")
    by_book = merged.groupby("book")[["p_prop", "p_shin", "actual_scorers"]].mean().sort_values("actual_scorers")
    print(by_book.to_string())


def report_calibration(td: pd.DataFrame, actual_scorer_pairs: pd.DataFrame,
                        participation: dict) -> None:
    print("\n" + "=" * 70)
    print("PART 5 — Calibration: predicted de-vigged probability vs actual hit rate")
    print("=" * 70)

    scored = set(zip(actual_scorer_pairs.game_id, actual_scorer_pairs.player_id))

    def is_scratch(row):
        participants = participation.get(row.game_id)
        return participants is not None and row.player_id not in participants

    is_scratch_mask = td.apply(is_scratch, axis=1)
    n_scratch = int(is_scratch_mask.sum())
    print(f"\nProp rows whose player never appears in that game's pbp at all (scratches, excluded): "
          f"{n_scratch} / {len(td)} ({n_scratch / len(td) * 100:.2f}%)")

    df = td[~is_scratch_mask].copy()
    df["actual"] = df.apply(lambda r: (r.game_id, r.player_id) in scored, axis=1)

    def calibration_table(sub: pd.DataFrame, prob_col: str) -> pd.DataFrame:
        sub = sub.copy()
        sub["decile"] = pd.qcut(sub[prob_col], q=10, duplicates="drop")
        table = sub.groupby("decile", observed=True).agg(
            n=("actual", "size"), predicted=(prob_col, "mean"), actual_hit_rate=("actual", "mean"))
        return table

    for prob_col, label in [("p_prop", "Proportional"), ("p_shin", "Shin")]:
        print(f"\n--- {label} de-vig, pooled across books, deciles ---")
        print(calibration_table(df, prob_col).to_string())

    for position in SKILL_POSITIONS:
        pos_df = df[df.position == position]
        print(f"\n--- {position} only, proportional de-vig, deciles (n={len(pos_df)}) ---")
        print(calibration_table(pos_df, "p_prop").to_string())
        print(f"\n--- {position} only, Shin de-vig, deciles (n={len(pos_df)}) ---")
        print(calibration_table(pos_df, "p_shin").to_string())


def main() -> None:
    td = load_td_props()
    print(f"Loaded {len(td)} anytime-TD Yes-side QB/RB/WR/TE prop rows")
    td = compute_devig(td)

    report_overround(td)
    report_devig(td)

    scorers = compute_actual_scorers()
    per_game_actual = report_actual_scorers(scorers)

    report_comparison(td, per_game_actual)

    participation = compute_participation_sets()
    report_calibration(td, scorers, participation)


if __name__ == "__main__":
    main()
