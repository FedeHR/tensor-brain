"""Tables and figures for the noisy-perception (E1) and volatility (E2) grids.

Both grids inherit the gridworld's bimodal outcome, so every quantity is
reported the same way as in the earlier study: an escape rate, and metrics
conditioned on the seeds that escaped.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.agency.analysis import ESCAPE_THRESHOLD  # noqa: E402
from experiments.agency.plots import condition_style  # noqa: E402

E1_CONDITIONS = ("tb-full", "gru-control", "lstm-control", "decoupled-feedback")
E2_CONDITIONS = ("alpha-1.0", "alpha-0.5", "alpha-0.0", "alpha-learned")
ALPHA_OF = {"alpha-0.0": 0.0, "alpha-0.5": 0.5, "alpha-1.0": 1.0}


def load(root: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(root.glob("*/*/seed*/result.json"))]


def _escaped(run: dict) -> bool:
    return run["final"]["success_rate"] > ESCAPE_THRESHOLD


def summarize(runs: list[dict], key, metric: str) -> dict:
    """Mean, s.e. and escape rate per group, conditioned on escaping."""

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[key(run)].append(run)
    summary = {}
    for group, entries in grouped.items():
        escaped = [entry for entry in entries if _escaped(entry)]
        values = [
            entry["final"].get(metric, entry.get("calibration", {}).get(metric, np.nan))
            for entry in escaped
        ]
        values = [value for value in values if value == value]
        summary[group] = {
            "mean": float(np.mean(values)) if values else float("nan"),
            "sem": (
                float(np.std(values) / max(1.0, np.sqrt(len(values))))
                if values
                else float("nan")
            ),
            "escaped": len(escaped),
            "seeds": len(entries),
        }
    return summary


def e1_figure(runs: list[dict], path: Path) -> str:
    """First-choice accuracy against observation noise, per architecture."""

    runs = [run for run in runs if run["hazard"] == 0.0 and run["condition"] in E1_CONDITIONS]
    noises = sorted({run["noise"] for run in runs})
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    lines = ["| condition | " + " | ".join(f"eps={n:g}" for n in noises) + " |",
             "|---" * (len(noises) + 1) + "|"]
    for position, condition in enumerate(E1_CONDITIONS):
        style = condition_style(condition, position)
        for axis, metric in zip(axes, ("first_choice_accuracy", "belief_correlation"), strict=True):
            summary = summarize(
                [r for r in runs if r["condition"] == condition],
                lambda r: r["noise"], metric,
            )
            xs = [n for n in noises if n in summary]
            ys = [summary[n]["mean"] for n in xs]
            es = [summary[n]["sem"] for n in xs]
            axis.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=condition, **style)
        summary = summarize(
            [r for r in runs if r["condition"] == condition],
            lambda r: r["noise"], "first_choice_accuracy",
        )
        lines.append(
            f"| `{condition}` | "
            + " | ".join(
                f"{summary[n]['mean']:.3f} ± {summary[n]['sem']:.3f} "
                f"({summary[n]['escaped']}/{summary[n]['seeds']})"
                if n in summary
                else "-"
                for n in noises
            )
            + " |"
        )
    axes[0].axhline(1 / 3, color="black", linestyle=":", linewidth=1)
    axes[0].text(noises[-1], 1 / 3 + 0.01, "cue-blind chance", fontsize=7, ha="right")
    axes[0].set_ylabel("first-choice accuracy")
    axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1)
    axes[1].set_ylabel("corr(P(collect), exact Bayes posterior)")
    for axis in axes:
        axis.set_xlabel("observation noise $\\epsilon$")
        axis.grid(alpha=0.25)
    axes[0].set_title("E1: does the advantage grow with noise?")
    axes[1].set_title("Is the agent's commitment Bayes-calibrated?")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return "\n".join(lines) + "\n"


def e2_figure(runs: list[dict], path: Path) -> str:
    """Performance against the prior weight alpha, per volatility."""

    runs = [run for run in runs if run["condition"] in E2_CONDITIONS]
    hazards = sorted({run["hazard"] for run in runs})
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(hazards)))
    lines = ["| hazard | " + " | ".join(f"`{c}`" for c in E2_CONDITIONS) + " |",
             "|---" * (len(E2_CONDITIONS) + 1) + "|"]
    for colour, hazard in zip(palette, hazards, strict=True):
        subset = [r for r in runs if r["hazard"] == hazard]
        fixed = summarize(
            [r for r in subset if r["condition"] in ALPHA_OF],
            lambda r: ALPHA_OF[r["condition"]], "first_choice_accuracy",
        )
        xs = sorted(fixed)
        axes[0].errorbar(
            xs, [fixed[a]["mean"] for a in xs], yerr=[fixed[a]["sem"] for a in xs],
            marker="o", capsize=3, color=colour, label=f"h = {hazard:g}",
        )
        learned = [
            r["learned_alpha"] for r in subset
            if r["condition"] == "alpha-learned" and _escaped(r) and r["learned_alpha"] is not None
        ]
        if learned:
            axes[1].errorbar(
                [hazard], [np.mean(learned)],
                yerr=[np.std(learned) / max(1.0, np.sqrt(len(learned)))],
                marker="s", capsize=4, color=colour, markersize=9,
            )
        row = summarize(subset, lambda r: r["condition"], "first_choice_accuracy")
        lines.append(
            f"| {hazard:g} | "
            + " | ".join(
                f"{row[c]['mean']:.3f} ± {row[c]['sem']:.3f} "
                f"({row[c]['escaped']}/{row[c]['seeds']})"
                if c in row
                else "-"
                for c in E2_CONDITIONS
            )
            + " |"
        )
    axes[0].set_xlabel(r"prior weight $\alpha$ (0 = neural PVM, 1 = HB-POVM)")
    axes[0].set_ylabel("first-choice accuracy")
    axes[0].set_title(r"E2: which $\alpha$ does the volatility call for?")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("hazard rate $h$")
    axes[1].set_ylabel(r"learned $\alpha$")
    axes[1].set_title(r"Does a learned $\alpha$ track volatility?")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", type=Path, default=Path("runs/agency/noisy"))
    parser.add_argument("--figure-root", type=Path, default=Path("docs/figures/agency/noisy"))
    arguments = parser.parse_args()
    arguments.figure_root.mkdir(parents=True, exist_ok=True)

    runs = load(arguments.grid_root)
    if not runs:
        raise SystemExit(f"no results under {arguments.grid_root}")
    e1 = e1_figure(runs, arguments.figure_root / "e1_noise.png")
    e2 = e2_figure(runs, arguments.figure_root / "e2_volatility.png")
    table = (
        "### E1 - first-choice accuracy by observation noise (escaped seeds)\n\n" + e1
        + "\n### E2 - first-choice accuracy by prior weight and volatility\n\n" + e2
    )
    (arguments.figure_root / "summary_table.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
