"""Aggregation that respects the bimodal outcome of this task.

Runs on this environment do not vary smoothly with the seed. A run either
escapes the sparse-reward local optimum in which ``collect`` is suppressed and
every episode times out, or it never does. Escaped runs land near ceiling;
trapped runs sit at exactly zero success and a return of
``-max_steps * step_penalty``.

Averaging across that bimodality produces numbers that describe *how many seeds
escaped*, not how well the architecture behaves. Every reported quantity is
therefore split in two:

* **escape rate** -- the fraction of seeds that found the positive outcome at
  all, which is a real and comparable property of a condition; and
* **conditional metrics** -- performance among the seeds that escaped, which is
  what the ablation was actually asking about.

A condition that never escapes in any seed is reported as such rather than as a
low average.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from experiments.agency.conditions import CONDITIONS

# A trapped run has exactly zero successes; escaped runs are near ceiling. Any
# threshold in the wide empty middle gives the same partition.
ESCAPE_THRESHOLD = 0.10


@dataclass(frozen=True)
class ConditionSummary:
    """Escape rate plus metrics conditioned on having escaped."""

    condition: str
    seeds: int
    escaped: int
    metrics: dict[str, dict[str, float]]

    @property
    def escape_rate(self) -> float:
        return self.escaped / self.seeds if self.seeds else 0.0

    def value(self, split: str, metric: str) -> float | None:
        return self.metrics.get(f"{split}_{metric}", {}).get("mean")

    def error(self, split: str, metric: str) -> float:
        return self.metrics.get(f"{split}_{metric}", {}).get("sem", 0.0)


def _mean_sem(values: Sequence[float]) -> dict[str, float]:
    count = len(values)
    if count == 0:
        return {"mean": float("nan"), "sem": float("nan"), "count": 0}
    mean = sum(values) / count
    if count == 1:
        return {"mean": mean, "sem": 0.0, "count": 1}
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    return {"mean": mean, "sem": (variance / count) ** 0.5, "count": count}


def summarize_condition(
    condition: str,
    per_seed: dict[str, list[dict[str, float]]],
    *,
    metrics: Sequence[str],
) -> ConditionSummary:
    """Split one condition's seeds into escaped and trapped, then aggregate."""

    successes = [entry["success_rate"] for entry in per_seed["eval"]]
    escaped = [index for index, value in enumerate(successes) if value > ESCAPE_THRESHOLD]
    aggregated: dict[str, dict[str, float]] = {}
    for split, entries in per_seed.items():
        for metric in metrics:
            aggregated[f"{split}_{metric}"] = _mean_sem(
                [entries[index][metric] for index in escaped if metric in entries[index]]
            )
    return ConditionSummary(condition, len(successes), len(escaped), aggregated)


def load_summaries(
    reevaluation: Path, *, metrics: Sequence[str]
) -> dict[str, ConditionSummary]:
    """Summarize every condition present in a re-evaluation file."""

    scored = json.loads(reevaluation.read_text())
    return {
        condition: summarize_condition(condition, scored[condition], metrics=metrics)
        for condition in CONDITIONS
        if condition in scored
    }


def markdown_table(summaries: dict[str, ConditionSummary]) -> str:
    """Render the reported table: escape rate first, then conditional metrics."""

    header = (
        "| condition | escaped seeds | first choice, train cues | first choice, held-out cues "
        "| return | distractor/ep | steps |\n|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for name, summary in summaries.items():
        if summary.escaped == 0:
            rows.append(
                f"| `{name}` | 0 / {summary.seeds} | — | — | — | — | never escaped |"
            )
            continue
        rows.append(
            f"| `{name}` | {summary.escaped} / {summary.seeds} "
            f"| {summary.value('eval', 'first_choice_accuracy'):.3f} "
            f"± {summary.error('eval', 'first_choice_accuracy'):.3f} "
            f"| {summary.value('holdout', 'first_choice_accuracy'):.3f} "
            f"± {summary.error('holdout', 'first_choice_accuracy'):.3f} "
            f"| {summary.value('eval', 'mean_return'):+.3f} "
            f"| {summary.value('eval', 'distractor_rate'):.2f} "
            f"| {summary.value('eval', 'mean_length'):.1f} |"
        )
    return header + "\n".join(rows) + "\n"
