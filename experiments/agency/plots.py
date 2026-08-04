"""Figures for the agency experiments.

Quantitative figures answer whether a condition works; the qualitative ones show
*what the agent named and chose*, which is the part a scalar cannot carry.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.agency.diagnostics import NarratedEpisode  # noqa: E402
from experiments.agency.gridworld import ACTION_NAMES  # noqa: E402
from experiments.agency.vocabulary import COLOR_NAMES, SHAPE_NAMES  # noqa: E402

CONDITION_COLOR = {
    "tb-full": "#1f77b4",
    "gru-control": "#7f7f7f",
}
SHAPE_MARKER = {"key": "P", "ball": "o", "box": "s", "cup": "v", "star": "*", "ring": "D"}
COLOR_HEX = {
    "red": "#d62728",
    "green": "#2ca02c",
    "blue": "#1f77b4",
    "yellow": "#bcbd22",
    "purple": "#9467bd",
    "cyan": "#17becf",
}


def _series(
    runs: Sequence[Mapping], split: str, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and standard error over seeds of one logged metric."""

    episodes = np.asarray(runs[0]["log"]["episodes"], dtype=float)
    values = np.stack(
        [
            [point.get(metric, np.nan) for point in run["log"][f"{split}_metrics"]]
            for run in runs
        ]
    )
    mean = np.nanmean(values, axis=0)
    error = np.nanstd(values, axis=0) / max(1.0, np.sqrt(values.shape[0]))
    return episodes, mean, error


def learning_curves(
    results: Mapping[str, Sequence[Mapping]],
    path: Path,
    *,
    conditions: Sequence[str],
    metric: str = "success_rate",
    title: str = "Cued-object success during REINFORCE",
    limit: tuple[float, float] | None = (-0.02, 1.02),
) -> None:
    """Success rate against environment episodes, mean +- s.e. over seeds."""

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for split, axis in zip(("eval", "holdout"), axes, strict=True):
        for name in conditions:
            if name not in results:
                continue
            episodes, mean, error = _series(results[name], split, metric)
            colour = CONDITION_COLOR.get(name)
            axis.plot(episodes, mean, label=name, color=colour, linewidth=1.8)
            axis.fill_between(episodes, mean - error, mean + error, alpha=0.15, color=colour)
        axis.set_xlabel("training episodes")
        axis.grid(alpha=0.25)
        if limit is not None:
            axis.set_ylim(*limit)
    axes[0].set_title("training cue conjunctions")
    axes[1].set_title("held-out cue conjunctions (zero shot)")
    axes[0].set_ylabel(metric.replace("_", " "))
    axes[1].legend(fontsize=8, loc="upper left", framealpha=0.9)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def ablation_bars(
    finals: Mapping[str, Mapping[str, Mapping[str, list[float]]]],
    path: Path,
    *,
    conditions: Sequence[str],
    metric: str = "success_rate",
    title: str = "Final performance by condition",
    reference_line: float | None = 1 / 3,
    limit: tuple[float, float] | None = (0.0, 1.05),
) -> None:
    """Final metric per condition, training cues beside held-out cues."""

    names = [name for name in conditions if name in finals]
    positions = np.arange(len(names))
    width = 0.38
    figure, axis = plt.subplots(figsize=(max(7.0, 0.62 * len(names) + 3), 4.4))
    splits = ((-width / 2, "eval", "#1f77b4"), (width / 2, "holdout", "#ff7f0e"))
    for offset, split, colour in splits:
        values = np.array([np.mean(finals[name][split][metric]) for name in names])
        errors = np.array(
            [
                np.std(finals[name][split][metric])
                / max(1.0, np.sqrt(len(finals[name][split][metric])))
                for name in names
            ]
        )
        axis.bar(
            positions + offset, values, width,
            yerr=errors, capsize=3, label=split, color=colour,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    axis.set_ylabel(metric.replace("_", " "))
    if limit is not None:
        axis.set_ylim(*limit)
    if reference_line is not None:
        axis.axhline(reference_line, color="black", linestyle=":", linewidth=1)
        axis.text(
            len(names) - 0.5,
            reference_line + 0.015,
            "one-shot cue-blind chance",
            fontsize=7,
            ha="right",
        )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _draw_grid(axis, episode: NarratedEpisode, step: int) -> None:
    size = episode.grid.size
    axis.set_xlim(-0.5, size - 0.5)
    axis.set_ylim(size - 0.5, -0.5)
    axis.set_xticks(range(size))
    axis.set_yticks(range(size))
    axis.grid(color="#dddddd", linewidth=0.6)
    axis.tick_params(labelbottom=False, labelleft=False, length=0)
    radius = episode.grid.view_radius
    axis.add_patch(
        plt.Rectangle(
            (episode.agent_col[step] - radius - 0.5, episode.agent_row[step] - radius - 0.5),
            2 * radius + 1,
            2 * radius + 1,
            facecolor="#fff3cd",
            edgecolor="none",
            zorder=0,
        )
    )
    for slot, (row, column) in enumerate(zip(episode.object_row, episode.object_col, strict=True)):
        colour = COLOR_HEX[COLOR_NAMES[episode.object_color[slot]]]
        marker = SHAPE_MARKER[SHAPE_NAMES[episode.object_shape[slot]]]
        axis.plot(
            column,
            row,
            marker=marker,
            markersize=13,
            color=colour,
            markeredgecolor="black" if slot == episode.target_slot else "none",
            markeredgewidth=1.8,
            zorder=2,
        )
    axis.plot(
        episode.agent_col[step],
        episode.agent_row[step],
        marker="X",
        markersize=11,
        color="black",
        zorder=3,
    )


def trajectory_strip(episode: NarratedEpisode, path: Path, *, max_steps: int = 10) -> None:
    """Grid renderings annotated with the agent's own symbolic narration."""

    steps = list(range(min(max_steps, len(episode.agent_row))))
    figure, axes = plt.subplots(1, len(steps), figsize=(1.75 * len(steps), 2.9))
    if len(steps) == 1:
        axes = [axes]
    for axis, step in zip(axes, steps, strict=True):
        _draw_grid(axis, episode, step)
        named = (
            f"sees: {episode.named_color[step].replace('nothing_visible', '-')}/"
            f"{episode.named_shape[step].replace('nothing_visible', '-')}"
            if episode.named_color
            else "sees: (not measured)"
        )
        axis.set_title(
            f"t={step}\n{named}\ndoes: {episode.action_name[step].replace('move_', '')}"
            f"\nv={episode.value[step]:+.2f}",
            fontsize=7,
        )
    figure.suptitle(
        f"Instruction: find the {episode.cue[0]} {episode.cue[1]}   "
        f"(black outline = target, shaded = field of view, X = agent)   "
        f"outcome: {'success' if episode.success else 'no target collected'}",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def index_rasters(episode: NarratedEpisode, path: Path) -> None:
    """Action and perceptual index distributions over one episode."""

    action = np.asarray(episode.action_probabilities).T
    has_percepts = bool(episode.percept_color_probabilities)
    rows = 3 if has_percepts else 2
    figure, axes = plt.subplots(
        rows, 1, figsize=(max(6.0, 0.42 * action.shape[1] + 2), 1.6 * rows + 1.2), sharex=True
    )
    axes[0].imshow(action, aspect="auto", cmap="magma", vmin=0, vmax=1)
    axes[0].set_yticks(range(len(ACTION_NAMES)))
    axes[0].set_yticklabels(ACTION_NAMES, fontsize=7)
    axes[0].set_ylabel("action index")
    if has_percepts:
        percept = np.asarray(episode.percept_color_probabilities).T
        labels = [*COLOR_NAMES[: percept.shape[0] - 1], "nothing"]
        axes[1].imshow(percept, aspect="auto", cmap="magma", vmin=0, vmax=1)
        axes[1].set_yticks(range(percept.shape[0]))
        axes[1].set_yticklabels(labels, fontsize=7)
        axes[1].set_ylabel("colour index")
    axes[-1].plot(episode.value, color="#1f77b4", label="reward-index value")
    axes[-1].plot(episode.reward, color="#d62728", linewidth=0.9, label="reward")
    axes[-1].legend(fontsize=7)
    axes[-1].grid(alpha=0.25)
    axes[-1].set_xlabel("environment step")
    figure.suptitle(
        f"Index measurement distributions, instruction: {episode.cue[0]} {episode.cue[1]}",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def similarity_heatmap(
    matrix: np.ndarray, labels: Sequence[str], path: Path, title: str
) -> None:
    """Cosine similarity between index embedding columns of ``A``."""

    figure, axis = plt.subplots(figsize=(0.42 * len(labels) + 3, 0.42 * len(labels) + 2.4))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=90, fontsize=7)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046)
    axis.set_title(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def cue_action_alignment(
    scores: np.ndarray, cue_labels: Sequence[str], action_labels: Sequence[str], path: Path
) -> None:
    """How strongly each cue embedding alone excites each action index."""

    figure, axis = plt.subplots(
        figsize=(0.8 * len(action_labels) + 3, 0.42 * len(cue_labels) + 2.2)
    )
    limit = float(np.abs(scores).max()) or 1.0
    image = axis.imshow(scores, cmap="PuOr_r", vmin=-limit, vmax=limit)
    axis.set_xticks(range(len(action_labels)))
    axis.set_xticklabels(action_labels, rotation=30, ha="right", fontsize=8)
    axis.set_yticks(range(len(cue_labels)))
    axis.set_yticklabels(cue_labels, fontsize=8)
    figure.colorbar(image, ax=axis, fraction=0.046)
    axis.set_title(
        r"action score from a pure cue state: $a_{action}^\top\sigma(a_{cue})$", fontsize=10
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def value_map(landscape: np.ndarray, episode: NarratedEpisode, path: Path) -> None:
    """The reward index's score across a frozen layout."""

    figure, axis = plt.subplots(figsize=(4.6, 4.2))
    image = axis.imshow(landscape, cmap="viridis")
    for slot, (row, column) in enumerate(zip(episode.object_row, episode.object_col, strict=True)):
        axis.plot(
            column,
            row,
            marker=SHAPE_MARKER[SHAPE_NAMES[episode.object_shape[slot]]],
            markersize=15,
            color=COLOR_HEX[COLOR_NAMES[episode.object_color[slot]]],
            markeredgecolor="white" if slot == episode.target_slot else "none",
            markeredgewidth=2.2,
        )
    figure.colorbar(image, ax=axis, fraction=0.046)
    axis.set_xticks(range(landscape.shape[1]))
    axis.set_yticks(range(landscape.shape[0]))
    axis.set_title(
        f"reward-index value $a_r^\\top\\sigma(q)$ per agent cell\ninstruction: "
        f"{episode.cue[0]} {episode.cue[1]} (white outline)",
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def load_results(root: Path, condition: str) -> list[dict]:
    """Load every seed of one condition from a grid output directory."""

    runs = []
    for path in sorted((root / condition).glob("seed*/result.json")):
        runs.append(json.loads(path.read_text()))
    return runs


def escape_and_conditional(
    summaries: Mapping[str, object], path: Path, *, conditions: Sequence[str]
) -> None:
    """Escape rate beside first-choice accuracy among the seeds that escaped.

    Two panels rather than one number, because averaging a bimodal outcome
    reports how many seeds escaped rather than how the architecture behaves.
    """

    names = [name for name in conditions if name in summaries]
    positions = np.arange(len(names))
    figure, axes = plt.subplots(
        2, 1, figsize=(max(7.0, 0.62 * len(names) + 3), 7.6), sharex=True
    )
    axes[0].bar(
        positions,
        [summaries[name].escape_rate for name in names],
        0.62,
        color="#4c72b0",
    )
    axes[0].set_ylabel("fraction of seeds that\nescaped the never-collect optimum")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_title("Reliability: did the run find the positive outcome at all?")

    width = 0.38
    splits = ((-width / 2, "eval", "#1f77b4"), (width / 2, "holdout", "#ff7f0e"))
    for offset, split, colour in splits:
        values, errors = [], []
        for name in names:
            value = summaries[name].value(split, "first_choice_accuracy")
            values.append(0.0 if value is None or value != value else value)
            errors.append(summaries[name].error(split, "first_choice_accuracy"))
        axes[1].bar(
            positions + offset, values, width,
            yerr=errors, capsize=3, label=f"{split} cues", color=colour,
        )
    axes[1].axhline(1 / 3, color="black", linestyle=":", linewidth=1)
    axes[1].text(len(names) - 0.5, 1 / 3 + 0.02, "cue-blind chance", fontsize=7, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("first-choice accuracy\n(escaped seeds only)")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    axes[1].set_title("Instruction following, conditioned on having escaped")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
