"""Build every reported figure and the summary table from a finished grid.

Usage::

    python -m experiments.agency.run_figures \
        --grid-root runs/agency/grid --figure-root docs/figures/agency
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.agency.agent import GridAgent
from experiments.agency.analysis import load_summaries, markdown_table
from experiments.agency.conditions import CLAIM_GROUPS, CONDITIONS, TASK
from experiments.agency.diagnostics import (
    action_alignment,
    index_similarity,
    narrate_episode,
    value_landscape,
)
from experiments.agency.gridworld import (
    SymbolicForaging,
    latin_square_holdout,
    train_cues,
)
from experiments.agency.plots import (
    ablation_bars,
    cue_action_alignment,
    escape_and_conditional,
    index_rasters,
    learning_curves,
    load_results,
    similarity_heatmap,
    trajectory_strip,
    value_map,
)

# Success rate alone is not sufficient on this task variant. Because collecting
# a distractor is penalised but not terminal, a cue-blind agent can brute-force
# the episode by collecting objects until the reward turns positive. `no-cue`
# does exactly that on some seeds. `mean_return` and `distractor_rate` are what
# separate an agent that follows the instruction from one that does not.
REPORTED_METRICS = (
    "success_rate",
    "mean_return",
    "distractor_rate",
    "mean_length",
    "percept_accuracy",
    "first_choice_accuracy",
    "first_choice_rate",
)


def collect(
    grid_root: Path, reevaluation: Path | None = None
) -> tuple[dict[str, list[dict]], dict[str, dict[str, dict[str, list[float]]]]]:
    """Load every condition's seeds and reduce their final metrics.

    When a re-evaluation file is present its metrics replace the ones recorded
    during training, so that every condition is scored under identical, final
    metric definitions rather than whatever existed when its run started.
    """

    scored = json.loads(reevaluation.read_text()) if reevaluation else {}
    results: dict[str, list[dict]] = {}
    finals: dict[str, dict[str, dict[str, list[float]]]] = {}
    for condition in CONDITIONS:
        runs = load_results(grid_root, condition)
        if not runs:
            continue
        results[condition] = runs
        entries = scored.get(condition)
        finals[condition] = {
            split: {
                metric: [
                    entry[metric]
                    for entry in (
                        entries[split] if entries else [run["final"][split] for run in runs]
                    )
                    if metric in entry
                ]
                for metric in REPORTED_METRICS
            }
            for split in ("eval", "holdout")
        }
    return results, finals


def write_summary(finals: dict, results: dict, path: Path) -> str:
    """Write the machine-readable summary and return a Markdown table."""

    payload = {
        condition: {
            "seeds": len(results[condition]),
            "num_parameters": results[condition][0]["num_parameters"],
            "seconds": float(np.mean([run["seconds"] for run in results[condition]])),
            **{
                f"{split}_{metric}": {
                    "mean": float(np.mean(values)),
                    "sem": float(np.std(values) / max(1.0, np.sqrt(len(values)))),
                    "values": values,
                }
                for split, metrics in finals[condition].items()
                for metric, values in metrics.items()
            },
        }
        for condition in finals
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    header = (
        "| condition | params | first choice (train cues) | first choice (held-out cues) "
        "| return | success | distractor/ep | steps |\n|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for condition, values in payload.items():
        train = values["eval_first_choice_accuracy"]
        held = values["holdout_first_choice_accuracy"]
        rows.append(
            f"| `{condition}` | {values['num_parameters']:,} "
            f"| {train['mean']:.3f} ± {train['sem']:.3f} "
            f"| {held['mean']:.3f} ± {held['sem']:.3f} "
            f"| {values['eval_mean_return']['mean']:+.3f} "
            f"| {values['eval_success_rate']['mean']:.3f} "
            f"| {values['eval_distractor_rate']['mean']:.2f} "
            f"| {values['eval_mean_length']['mean']:.1f} |"
        )
    return header + "\n".join(rows) + "\n"


def qualitative_figures(
    grid_root: Path, figure_root: Path, *, condition: str, seed: int, prefix: str = ""
) -> None:
    """Regenerate the narration, raster, geometry and value figures."""

    agent_config = CONDITIONS[condition]
    assert agent_config is not None, "qualitative figures require a Tensor Brain agent"
    checkpoint = grid_root / condition / f"seed{seed}" / "checkpoint.pt"
    agent = GridAgent(TASK, agent_config)
    agent.load_state_dict(torch.load(checkpoint, weights_only=True)["model_state_dict"])
    agent.eval()

    for split, cues in (
        ("train-cue", train_cues(TASK.num_colors, TASK.num_shapes)),
        ("holdout-cue", latin_square_holdout(TASK.num_colors, TASK.num_shapes)),
    ):
        # Keep the *shortest successful* episode out of a sample, so the strip
        # fits on a page and its panels actually show the agent reaching the
        # target rather than being truncated before the outcome.
        environment = SymbolicForaging(TASK, 1, seed=7, allowed_cues=cues)
        candidates = [narrate_episode(environment, agent) for _ in range(30)]
        successful = [item for item in candidates if item.success]
        episode = min(successful or candidates, key=lambda item: len(item.agent_row))
        trajectory_strip(episode, figure_root / f"{prefix}trajectory_{split}.png")
        index_rasters(episode, figure_root / f"{prefix}rasters_{split}.png")
        if split == "train-cue":
            landscape = value_landscape(environment, agent)
            value_map(
                landscape.numpy(), episode, figure_root / f"{prefix}value_landscape.png"
            )

    similarity, labels = index_similarity(agent)
    similarity_heatmap(
        similarity.numpy(),
        labels,
        figure_root / f"{prefix}index_geometry.png",
        "cosine similarity between columns of the shared index matrix $A$",
    )
    scores, cue_labels, action_labels = action_alignment(agent)
    cue_action_alignment(
        scores.numpy(),
        cue_labels,
        action_labels,
        figure_root / f"{prefix}cue_action_alignment.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", type=Path, default=Path("runs/agency/grid"))
    parser.add_argument("--figure-root", type=Path, default=Path("docs/figures/agency"))
    parser.add_argument("--reevaluation", type=Path, default=Path("runs/agency/reevaluation.json"))
    parser.add_argument("--qualitative-condition", default="tb-full")
    parser.add_argument("--qualitative-seed", type=int, default=0)
    arguments = parser.parse_args()
    arguments.figure_root.mkdir(parents=True, exist_ok=True)

    reevaluation = arguments.reevaluation if arguments.reevaluation.exists() else None
    results, finals = collect(arguments.grid_root, reevaluation)
    if not results:
        raise SystemExit(f"no results found under {arguments.grid_root}")

    table = write_summary(finals, results, arguments.figure_root / "summary.json")
    (arguments.figure_root / "summary_table.md").write_text(table)

    if reevaluation is not None:
        summaries = load_summaries(reevaluation, metrics=REPORTED_METRICS)
        escape_table = markdown_table(summaries)
        (arguments.figure_root / "escape_table.md").write_text(escape_table)
        print(escape_table)
        escape_and_conditional(
            summaries,
            arguments.figure_root / "escape_and_conditional.png",
            conditions=list(summaries),
        )
    else:
        print(table)

    learning_curves(
        results,
        arguments.figure_root / "learning_curves_main.png",
        conditions=("tb-full", "gru-control", "no-cue", "deliberate-3-attend"),
        title="Reference agent, control policy, and the cue-blind floor",
    )
    ablation_bars(
        finals,
        arguments.figure_root / "ablation_first_choice.png",
        conditions=list(finals),
        metric="first_choice_accuracy",
        title="Did the agent's first choice match the instruction? (3 seeds, 320k episodes)",
    )
    ablation_bars(
        finals,
        arguments.figure_root / "ablation_success.png",
        conditions=list(finals),
        title="Final cued-object success after 320k episodes (3 seeds)",
    )
    ablation_bars(
        finals,
        arguments.figure_root / "ablation_distractor.png",
        conditions=list(finals),
        metric="distractor_rate",
        title="Distractor collections per episode: brute forcing versus following the cue",
        reference_line=None,
        limit=None,
    )
    ablation_bars(
        finals,
        arguments.figure_root / "ablation_return.png",
        conditions=list(finals),
        metric="mean_return",
        title="Mean episode return (nets out brute-force distractor collection)",
        reference_line=None,
        limit=None,
    )
    for claim, conditions in CLAIM_GROUPS.items():
        stem = claim.split()[0].lower()
        learning_curves(
            results,
            arguments.figure_root / f"curves_{stem}_return.png",
            conditions=conditions,
            metric="mean_return",
            title=f"{claim} - episode return",
            limit=None,
        )
    for claim, conditions in CLAIM_GROUPS.items():
        stem = claim.split()[0].lower()
        learning_curves(
            results,
            arguments.figure_root / f"curves_{stem}.png",
            conditions=conditions,
            title=claim,
        )
    # The reference agent, and the condition that follows the instruction best.
    for condition, prefix in (
        (arguments.qualitative_condition, "tbfull_"),
        ("deliberate-3-attend", "deliberate3_"),
    ):
        qualitative_figures(
            arguments.grid_root,
            arguments.figure_root,
            condition=condition,
            seed=arguments.qualitative_seed,
            prefix=prefix,
        )


if __name__ == "__main__":
    main()
