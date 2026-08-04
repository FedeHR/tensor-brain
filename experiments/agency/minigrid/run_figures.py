"""Build every reported figure and table for the MiniGrid study.

Usage::

    python -m experiments.agency.minigrid.run_figures \
        --grid-root runs/agency/minigrid --figure-root docs/figures/agency/minigrid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.agency.minigrid.conditions import CONDITIONS, LEVEL_CONDITIONS, LEVELS
from experiments.agency.minigrid.diagnostics import best_episode
from experiments.agency.minigrid.env import VectorMiniGrid
from experiments.agency.minigrid.plots import (
    filmstrip,
    final_bars,
    index_raster,
    learning_curves,
    trajectory_overlay,
)
from experiments.agency.minigrid.run import build_policy, cue_split

# Measured with a uniform random policy over several hundred episodes.
RANDOM_SUCCESS = {"gotolocal": 0.31, "doorkey": 0.02, "pickupstrict": 0.10}


def load(grid_root: Path, level: str, condition: str) -> list[dict]:
    return [
        json.loads(path.read_text())
        for path in sorted((grid_root / level / condition).glob("seed*/result.json"))
    ]


def collect(grid_root: Path, level: str) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """Load every condition's seeds and reduce their final metrics."""

    results: dict[str, list[dict]] = {}
    finals: dict[str, dict[str, list[float]]] = {}
    for condition in LEVEL_CONDITIONS[level]:
        runs = load(grid_root, level, condition)
        if not runs:
            continue
        results[condition] = runs
        finals[condition] = {
            metric: [run["log"]["train_metrics"][-1].get(metric, float("nan")) for run in runs]
            for metric in ("success_rate", "mean_return", "mean_length")
        }
        for split in ("eval", "holdout"):
            points = [run["log"][f"{split}_metrics"][-1] for run in runs]
            if points and points[0]:
                finals[condition][f"{split}_success_rate"] = [
                    point.get("success_rate", float("nan")) for point in points
                ]
    return results, finals


def summary_table(level: str, finals: dict[str, dict]) -> str:
    """Markdown table of the final metrics for one level."""

    compositional = LEVELS[level].compositional
    header = "| condition | seeds | success | " + (
        "held-out missions | " if compositional else ""
    ) + "return | steps |\n|---|---|---|" + ("---|" if compositional else "") + "---|---|\n"
    rows = []
    for condition, metrics in finals.items():
        values = np.asarray(metrics["success_rate"], dtype=float)
        cell = f"{np.nanmean(values):.3f} ± {np.nanstd(values) / max(1, np.sqrt(len(values))):.3f}"
        held = ""
        if compositional and "holdout_success_rate" in metrics:
            other = np.asarray(metrics["holdout_success_rate"], dtype=float)
            held = (
                f" {np.nanmean(other):.3f} ± "
                f"{np.nanstd(other) / max(1, np.sqrt(len(other))):.3f} |"
            )
        rows.append(
            f"| `{condition}` | {len(values)} | {cell} |{held} "
            f"{np.nanmean(metrics['mean_return']):.3f} | "
            f"{np.nanmean(metrics['mean_length']):.1f} |"
        )
    return header + "\n".join(rows) + "\n"


def qualitative(grid_root: Path, figure_root: Path, level: str, condition: str, seed: int) -> None:
    """Rendered trajectory, filmstrip and index raster for one checkpoint."""

    specification = LEVELS[level]
    agent_config = CONDITIONS[condition]
    checkpoint = grid_root / level / condition / f"seed{seed}" / "checkpoint.pt"
    if not checkpoint.exists():
        return
    policy = build_policy(condition, agent_config)
    policy.load_state_dict(torch.load(checkpoint, weights_only=True)["model_state_dict"])
    policy.eval()
    train_cues, _ = cue_split(specification)
    environment = VectorMiniGrid(
        specification.env_id, 1, seed=17, allowed_cues=train_cues, render=True
    )
    stem = f"{level}_{condition}"
    # The trajectory figure wants the episode with the most to show; the
    # filmstrip wants one short enough to fit on a page.
    rich = best_episode(environment, policy, attempts=40, prefer="richest")
    trajectory_overlay(rich, figure_root / f"{stem}_trajectory.png")
    index_raster(
        rich,
        figure_root / f"{stem}_raster.png",
        list(policy.vocabulary.group_labels("action")),
    )
    brief = best_episode(environment, policy, attempts=20, prefer="shortest")
    filmstrip(brief, figure_root / f"{stem}_filmstrip.png")
    environment.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", type=Path, default=Path("runs/agency/minigrid"))
    parser.add_argument(
        "--figure-root", type=Path, default=Path("docs/figures/agency/minigrid")
    )
    parser.add_argument("--qualitative-seed", type=int, default=0)
    arguments = parser.parse_args()
    arguments.figure_root.mkdir(parents=True, exist_ok=True)

    tables = []
    for level, specification in LEVELS.items():
        results, finals = collect(arguments.grid_root, level)
        if not results:
            continue
        tables.append(f"### {level} (`{specification.env_id}`)\n\n{summary_table(level, finals)}")
        splits = ("train", "eval", "holdout") if specification.compositional else ("train",)
        learning_curves(
            results,
            arguments.figure_root / f"{level}_curves.png",
            conditions=list(results),
            splits=splits,
            title=f"{specification.env_id} - {specification.tests}",
            chance=RANDOM_SUCCESS.get(level),
        )
        # Reaching a *wrong* object costs nothing in GoTo levels, so a cue-blind
        # agent can sweep the room and still finish most episodes -- the same
        # structural loophole the gridworld study found. BabyAI's reward is
        # `1 - 0.9 * steps / max_steps` on success, so return separates an agent
        # that goes straight to the named object from one that searches.
        learning_curves(
            results,
            arguments.figure_root / f"{level}_return.png",
            conditions=list(results),
            splits=splits,
            metric="mean_return",
            title=f"{specification.env_id} - episode return (speed-weighted success)",
        )
        final_bars(
            finals,
            arguments.figure_root / f"{level}_final.png",
            conditions=list(results),
            title=f"{specification.env_id}: final success ({specification.frames:,} frames)",
            chance=RANDOM_SUCCESS.get(level),
        )
        for condition in ("tb-full", "deliberate-3-attend"):
            if condition in results:
                qualitative(
                    arguments.grid_root, arguments.figure_root, level, condition,
                    arguments.qualitative_seed,
                )

    table = "\n".join(tables)
    (arguments.figure_root / "summary_table.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
