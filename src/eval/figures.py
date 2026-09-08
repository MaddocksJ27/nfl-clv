"""Five publication-quality figures for the README. Every number here is
recomputed fresh from cached parquet files on each run — nothing is
hardcoded from a previous script's printed output. Each figure function
prints the underlying table it plots, so the PNG can be cross-checked
against README prose directly.

matplotlib only (no seaborn). Style: minimal chartjunk, colourblind-safe
(Okabe-Ito) palette, clean spines, n annotated wherever a panel
aggregates a sample.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "figures"
DPI = 150

# Okabe-Ito colourblind-safe palette
COLORS = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "purple": "#CC79A7", "skyblue": "#56B4E9",
    "yellow": "#F0E442", "black": "#000000",
}

plt.rcParams.update({
    "figure.dpi": DPI, "savefig.dpi": DPI,
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def savefig(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}\n")
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 1 — power_curve.png
# ---------------------------------------------------------------------

def figure_power_curve() -> None:
    from src.models.spread_7d import build_dataset, simulate_detection_power, walk_forward_logistic_ridge

    print("=" * 70)
    print("Figure 1 — power_curve.png")
    print("=" * 70)

    df = build_dataset()
    oof = walk_forward_logistic_ridge(df)
    n_study = len(oof)
    print(f"This-study n (spread_7d.py walk-forward OOF sample) = {n_study}")

    n_grid = np.unique(np.round(np.logspace(np.log10(100), np.log10(100_000), 45)).astype(int))
    hit_rates = [0.53, 0.54, 0.55]
    palette = [COLORS["blue"], COLORS["vermillion"], COLORS["green"]]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for p, color in zip(hit_rates, palette):
        power = np.array([simulate_detection_power(p, int(n)) for n in n_grid])
        ax.plot(n_grid, power, color=color, lw=2.2, label=f"true hit rate {p * 100:.0f}%")
        print(f"  hit_rate={p:.2f}: power(n={n_grid[0]})={power[0]:.3f}  "
              f"power(n={n_study})={np.interp(n_study, n_grid, power):.3f}  power(n={n_grid[-1]})={power[-1]:.3f}")

    ax.axhline(0.80, color=COLORS["black"], lw=1, ls="--", alpha=0.55)
    ax.text(110, 0.815, "80% power", fontsize=8.5, color=COLORS["black"], alpha=0.8)

    ax.axvline(n_study, color=COLORS["black"], lw=1.2, ls=":")
    ax.annotate(f"this study\nn={n_study}", xy=(n_study, 0.02), xytext=(n_study * 1.3, 0.08),
                fontsize=9, ha="left", arrowprops=dict(arrowstyle="-", lw=0.8, color=COLORS["black"]))

    ax.set_xscale("log")
    ax.set_xlim(100, 100_000)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Number of bets (n, log scale)")
    ax.set_ylabel("P(bootstrap 5th percentile ROI > 0)")
    ax.set_title("Detection power vs. sample size — spread-direction betting at -110")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.65))

    savefig(fig, "power_curve.png")


# ---------------------------------------------------------------------
# Figure 2 — foresight_ceiling.png
# ---------------------------------------------------------------------

def figure_foresight_ceiling() -> None:
    from src.eval.book_disagreement import bootstrap_roi
    from src.eval.spread_horizon_ceiling import HORIZONS, VIG_PRICE, analyze_horizon, build_base_data
    from src.eval.validate_featured import compute_fixed_horizon_open

    print("=" * 70)
    print("Figure 2 — foresight_ceiling.png")
    print("=" * 70)

    games, consensus, close, margins = build_base_data()
    n_total = games.game_id.nunique()

    rows = []
    for h in HORIZONS:
        merged = analyze_horizon(h, games, consensus, close, margins)
        n_covered = compute_fixed_horizon_open(consensus, games, h).game_id.nunique()

        # perfect-foresight ceiling bet, same construction as spread_horizon_ceiling.report_ceiling
        d = merged[merged.close_spread != merged.open_spread].copy()
        d["bettable_side"] = np.where(d.close_spread < d.open_spread, "home", "away")
        d["home_cover_margin"] = d.margin_home + d.open_spread
        d["outcome"] = np.select(
            [d.home_cover_margin == 0,
             (d.bettable_side == "home") & (d.home_cover_margin > 0),
             (d.bettable_side == "away") & (d.home_cover_margin < 0)],
            ["push", "win", "win"], default="loss")
        d["profit"] = np.where(d.outcome == "push", np.nan,
                                np.where(d.outcome == "win", VIG_PRICE - 1.0, -1.0))
        decided = d[d.outcome != "push"]
        profits = decided["profit"].to_numpy()
        roi = profits.mean() * 100
        p5, p95 = np.percentile(bootstrap_roi(profits) * 100, [5, 95])

        rows.append({"horizon": h, "n": len(merged), "n_decided": len(decided),
                     "coverage_pct": n_covered / n_total * 100, "roi": roi, "p5": p5, "p95": p95})
        print(f"  {h}d: n={len(merged)}  n_decided={len(decided)}  coverage={n_covered / n_total * 100:.1f}%  "
              f"ROI={roi:+.2f}%  [{p5:+.2f}%, {p95:+.2f}%]")

    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    yerr = np.vstack([table.roi - table.p5, table.p95 - table.roi])
    ax.errorbar(table.horizon, table.roi, yerr=yerr, fmt="o", color=COLORS["blue"],
                capsize=4, markersize=7, lw=1.6, ecolor=COLORS["blue"], elinewidth=1.3)
    ax.axhline(0, color=COLORS["black"], lw=1, alpha=0.6)

    for _, r in table.iterrows():
        ax.annotate(f"n={int(r.n_decided)}\ncov={r.coverage_pct:.0f}%", (r.horizon, r.p95),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    ax.set_xticks(list(HORIZONS))
    ax.invert_xaxis()
    ax.set_xlabel("Days before kickoff (horizon)")
    ax.set_ylabel("ROI at -110 (%)")
    ax.set_title("Perfect-foresight ceiling on spread CLV, by horizon")

    savefig(fig, "foresight_ceiling.png")


# ---------------------------------------------------------------------
# Figure 3 — td_calibration.png
# ---------------------------------------------------------------------

def figure_td_calibration() -> None:
    from src.eval.props_sanity import (
        SKILL_POSITIONS, compute_actual_scorers, compute_is_scratch, compute_margin_devig,
        fit_expected_scorers_walkforward, load_game_totals, load_td_props,
    )

    print("=" * 70)
    print("Figure 3 — td_calibration.png")
    print("=" * 70)

    td = load_td_props()
    scorers = compute_actual_scorers()
    per_game_actual = scorers.groupby(["season", "game_id"])["player_id"].nunique().rename(
        "actual_scorers").reset_index()
    game_totals = load_game_totals()
    expected_scorers = fit_expected_scorers_walkforward(per_game_actual, game_totals)
    td = compute_margin_devig(td, expected_scorers)
    td = td[~compute_is_scratch(td)].copy()

    scored = set(zip(scorers.game_id, scorers.player_id))
    td["actual"] = list(zip(td.game_id, td.player_id))
    td["actual"] = td["actual"].isin(scored)

    def calibration_table(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        sub["decile"] = pd.qcut(sub["p_devig"], q=10, duplicates="drop")
        return sub.groupby("decile", observed=True).agg(
            n=("actual", "size"), predicted=("p_devig", "mean"), actual_hit_rate=("actual", "mean")
        ).reset_index(drop=True)

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), sharex=True, sharey=True)
    panels = [("Pooled", td)] + [(pos, td[td.position == pos]) for pos in SKILL_POSITIONS]

    axis_max = 0.0
    tables = {}
    for label, sub in panels:
        t = calibration_table(sub)
        tables[label] = t
        axis_max = max(axis_max, t.predicted.max(), t.actual_hit_rate.max())
    axis_max *= 1.1

    for ax, (label, sub) in zip(axes, panels):
        t = tables[label]
        sizes = 25 + 350 * (t.n / t.n.max())
        ax.scatter(t.predicted, t.actual_hit_rate, s=sizes, color=COLORS["blue"], alpha=0.85,
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.plot([0, axis_max], [0, axis_max], color=COLORS["black"], lw=1, ls="--", alpha=0.55, zorder=1)
        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.set_title(f"{label} (n={t.n.sum()})", fontsize=9.5)
        ax.set_xlabel("Predicted P(TD)")
        print(f"\n{label}: n={t.n.sum()}")
        print(t.round(4).to_string(index=False))

    axes[0].set_ylabel("Actual hit rate")
    fig.suptitle("Anytime-TD calibration by de-vigged-probability decile", y=1.04)

    savefig(fig, "td_calibration.png")


# ---------------------------------------------------------------------
# Figure 4 — drift_definitions.png
# ---------------------------------------------------------------------

def figure_drift_definitions() -> None:
    from src.eval.validate_featured import (
        build_event_schedule_map, compute_close_spreads, compute_consensus_home_spreads,
        compute_fixed_horizon_open, load_data, part2_drift_table,
    )

    print("=" * 70)
    print("Figure 4 — drift_definitions.png")
    print("=" * 70)

    featured, schedules = load_data()
    event_map = build_event_schedule_map(featured, schedules)
    games = event_map.drop_duplicates(subset="game_id")
    consensus = compute_consensus_home_spreads(featured, event_map)
    close = compute_close_spreads(consensus, games)

    print("\n--- first-observed drift (validate_featured.compute_close_spreads / part2_drift_table) ---")
    first_obs = part2_drift_table(consensus, games)

    definitions = {"first-observed\n(loose)": first_obs["drift"]}
    for h in (7, 3):
        horizon = compute_fixed_horizon_open(consensus, games, h)
        merged = horizon.merge(close[["game_id", "close_spread"]], on="game_id")
        merged["drift"] = merged.close_spread - merged.open_spread
        definitions[f"{h}-day horizon"] = merged["drift"]
        print(f"{h}d horizon drift: n={len(merged)}  mean={merged.drift.mean():+.3f}  sd={merged.drift.std():.3f}")

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.5), sharex=True)
    colors = [COLORS["blue"], COLORS["vermillion"], COLORS["green"]]
    bins = np.arange(-15.5, 15.6, 1)

    for ax, (label, series), color in zip(axes, definitions.items(), colors):
        n, sd = len(series), float(series.std())
        ax.hist(series.clip(-15, 15), bins=bins, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(0, color=COLORS["black"], lw=1, alpha=0.5)
        ax.set_ylabel(label.replace("\n", " "), fontsize=9.5)
        ax.text(0.02, 0.85, f"n={n}\nSD={sd:.2f}", transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", fc="white", ec="0.75", alpha=0.9))

    axes[-1].set_xlabel("Drift = close spread - open spread (points, home perspective; clipped to ±15 for display)")
    fig.suptitle("Open-to-close spread drift under three horizon definitions")

    savefig(fig, "drift_definitions.png")


# ---------------------------------------------------------------------
# Figure 5 — prop_skew.png
# ---------------------------------------------------------------------

def figure_prop_skew() -> None:
    from src.eval.book_disagreement import load_ou_props
    from src.eval.line_skew import OU_MARKETS, build_consensus_dataset, make_buckets

    print("=" * 70)
    print("Figure 5 — prop_skew.png")
    print("=" * 70)

    ou = load_ou_props()

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6), gridspec_kw={"hspace": 0.4})
    axes = axes.flatten()

    for i, (ax, market) in enumerate(zip(axes, OU_MARKETS)):
        df = build_consensus_dataset(ou, market)
        df["bucket"] = make_buckets(df)
        table = df.groupby("bucket", observed=True).agg(
            n=("actual", "size"), consensus_line=("consensus_line", "mean"),
            actual_mean=("actual", "mean"), actual_median=("actual", "median"),
        ).sort_values("consensus_line")

        ax.plot(table.consensus_line, table.consensus_line, color=COLORS["black"], lw=1, ls="--",
                alpha=0.55, label="book line" if i == 0 else None)
        ax.plot(table.consensus_line, table.actual_mean, color=COLORS["vermillion"], lw=2,
                marker="o", markersize=4, label="conditional mean" if i == 0 else None)
        ax.plot(table.consensus_line, table.actual_median, color=COLORS["blue"], lw=2,
                marker="s", markersize=4, label="conditional median" if i == 0 else None)

        ax.set_title(f"{market}  (n={int(table.n.sum())})", fontsize=9.5)
        ax.set_xlabel("Consensus line")
        ax.set_ylabel("Actual outcome")
        print(f"\n{market}: n={int(table.n.sum())}")
        print(table.round(3).to_string())

    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.97, 0.06), fontsize=10)
    fig.suptitle("Book line vs. conditional mean/median of actual outcomes, by prop market", y=1.02)

    savefig(fig, "prop_skew.png")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_power_curve()
    figure_foresight_ceiling()
    figure_td_calibration()
    figure_drift_definitions()
    figure_prop_skew()
    print("All 5 figures written to", FIGURES_DIR)


if __name__ == "__main__":
    main()
