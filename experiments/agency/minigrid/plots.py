"""Figures for the MiniGrid study."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from experiments.agency.minigrid.diagnostics import TILE_PIXELS, NarratedEpisode  # noqa: E402
from experiments.agency.plots import condition_style  # noqa: E402


def _series(runs: Sequence[Mapping], split: str, metric: str) -> tuple[np.ndarray, ...]:
    frames = np.asarray(runs[0]["log"]["frames"], dtype=float)
    values = np.stack(
        [
            [point.get(metric, np.nan) for point in run["log"][f"{split}_metrics"]]
            for run in runs
        ]
    )
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(values, axis=0)
        error = np.nanstd(values, axis=0) / max(1.0, np.sqrt(values.shape[0]))
    return frames, mean, error


def learning_curves(
    results: Mapping[str, Sequence[Mapping]],
    path: Path,
    *,
    conditions: Sequence[str],
    splits: Sequence[str] = ("train",),
    metric: str = "success_rate",
    title: str = "",
    chance: float | None = None,
) -> None:
    """Success against environment frames, mean +- s.e. over seeds."""

    figure, axes = plt.subplots(
        1, len(splits), figsize=(5.6 * len(splits) + 0.6, 4.2), sharey=True, squeeze=False
    )
    names = {"train": "training missions", "eval": "training missions (held-out layouts)",
             "holdout": "held-out mission combinations (zero shot)"}
    for split, axis in zip(splits, axes[0], strict=True):
        for position, name in enumerate(conditions):
            if name not in results:
                continue
            frames, mean, error = _series(results[name], split, metric)
            style = condition_style(name, position)
            axis.plot(frames, mean, label=name, **style)
            axis.fill_between(
                frames, mean - error, mean + error, alpha=0.13, color=style["color"]
            )
        if chance is not None:
            axis.axhline(chance, color="black", linestyle=":", linewidth=1)
            axis.text(
                frames[-1], chance + 0.015, "random policy", fontsize=7, ha="right"
            )
        axis.set_xlabel("environment frames")
        axis.set_title(names.get(split, split), fontsize=10)
        axis.grid(alpha=0.25)
        axis.set_ylim(-0.02, 1.02)
    axes[0][0].set_ylabel(metric.replace("_", " "))
    axes[0][-1].legend(fontsize=8, loc="lower right", framealpha=0.9)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def final_bars(
    finals: Mapping[str, Mapping[str, list[float]]],
    path: Path,
    *,
    conditions: Sequence[str],
    title: str,
    chance: float | None = None,
) -> None:
    """Final success per condition, with each seed shown as a point."""

    names = [name for name in conditions if name in finals]
    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(max(7.0, 0.75 * len(names) + 3), 4.4))
    for position, name in enumerate(names):
        values = np.asarray(finals[name]["success_rate"], dtype=float)
        style = condition_style(name, position)
        axis.bar(
            position,
            np.nanmean(values),
            0.62,
            yerr=np.nanstd(values) / max(1.0, np.sqrt(len(values))),
            capsize=3,
            color=style["color"],
            alpha=0.85,
        )
        axis.scatter(
            np.full(len(values), position), values, color="black", s=14, zorder=4
        )
    if chance is not None:
        axis.axhline(chance, color="black", linestyle=":", linewidth=1)
        axis.text(len(names) - 0.4, chance + 0.02, "random policy", fontsize=7, ha="right")
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axis.set_ylabel("success rate")
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", alpha=0.25)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def trajectory_overlay(episode: NarratedEpisode, path: Path) -> None:
    """The whole episode path drawn on the rendered map, coloured by time.

    Sub-goal events are annotated where they happened, which is what makes a
    DoorKey trajectory legible: fetch the key, cross to the door, open it, and
    only then head for the goal.
    """

    frame = episode.frames[min(episode.frames)]
    points = np.array(
        [[(x + 0.5) * TILE_PIXELS, (y + 0.5) * TILE_PIXELS] for x, y in episode.position]
    )
    figure, axis = plt.subplots(figsize=(6.4, 6.0))
    axis.imshow(frame)
    if len(points) > 1:
        segments = np.stack([points[:-1], points[1:]], axis=1)
        collection = LineCollection(
            segments, cmap="cool", linewidths=3.0, alpha=0.9,
            array=np.linspace(0, 1, len(segments)),
        )
        axis.add_collection(collection)
        bar = figure.colorbar(collection, ax=axis, fraction=0.046, pad=0.03)
        bar.set_label("episode progress", fontsize=8)
    axis.scatter(*points[0], marker="o", s=110, facecolor="white", edgecolor="black", zorder=5)
    axis.scatter(*points[-1], marker="*", s=260, facecolor="gold", edgecolor="black", zorder=5)
    for label, step in episode.events().items():
        axis.annotate(
            label,
            points[min(step, len(points) - 1)],
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=8,
            color="white",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.65},
        )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(
        f"{episode.env_id}\n\"{episode.mission}\"  -  "
        f"{'success' if episode.success else 'timeout'} in {episode.length} steps",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def filmstrip(episode: NarratedEpisode, path: Path, *, panels: int = 8) -> None:
    """Rendered frames at the episode's turning points, with the agent's symbols."""

    events = sorted(set(episode.events().values()) | {0, episode.length - 1})
    steps = sorted(episode.frames)
    chosen = list(events)
    for step in steps:
        if len(chosen) >= panels:
            break
        if all(abs(step - existing) > max(1, episode.length // panels) for existing in chosen):
            chosen.append(step)
    chosen = sorted(chosen)[:panels]

    figure, axes = plt.subplots(1, len(chosen), figsize=(2.0 * len(chosen), 3.0))
    axes = np.atleast_1d(axes)
    for axis, step in zip(axes, chosen, strict=True):
        axis.imshow(episode.frames[min(episode.frames, key=lambda k: abs(k - step))])
        axis.set_xticks([])
        axis.set_yticks([])
        named = (
            f"sees {episode.named_color[step]}/{episode.named_object[step]}".replace(
                "nothing_visible", "-"
            )
            if episode.named_color
            else "sees (not measured)"
        )
        axis.set_title(
            f"t={step}\n{named}\ndoes {episode.action_name[step]}\nv={episode.value[step]:+.2f}",
            fontsize=7,
        )
    figure.suptitle(f'"{episode.mission}"', fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def index_raster(episode: NarratedEpisode, path: Path, action_labels: Sequence[str]) -> None:
    """Action distribution and reward-index value over one episode."""

    action = np.asarray(episode.action_probabilities).T
    figure, axes = plt.subplots(
        2, 1, figsize=(max(6.0, 0.1 * action.shape[1] + 3), 5.0), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    image = axes[0].imshow(action, aspect="auto", cmap="magma", vmin=0, vmax=1)
    axes[0].set_yticks(range(len(action_labels)))
    axes[0].set_yticklabels(action_labels, fontsize=7)
    axes[0].set_ylabel("action index")
    figure.colorbar(image, ax=axes[0], fraction=0.03, pad=0.01)
    axes[1].plot(episode.value, color="#1f77b4", label="reward-index value")
    axes[1].plot(episode.reward, color="#d62728", linewidth=0.9, label="reward")
    for label, step in episode.events().items():
        axes[1].axvline(step, color="grey", linestyle="--", linewidth=0.8)
        axes[1].text(step, axes[1].get_ylim()[1], label, rotation=90, fontsize=6, va="top")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.25)
    axes[1].set_xlabel("environment step")
    figure.suptitle(f'Index measurements over one episode: "{episode.mission}"', fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=160)
    plt.close(figure)
