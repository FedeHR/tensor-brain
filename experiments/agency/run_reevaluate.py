"""Re-evaluate every saved checkpoint under the full metric set.

The grid's training loop logged the metrics that existed when it ran. This
script reloads each checkpoint and evaluates it again so that every condition is
scored on the same, final metric definitions -- in particular
``first_choice_accuracy``, which was added after the grid started.

That metric is the one that separates instruction following from brute force.
Because collecting a distractor is penalised but not terminal, an agent that
knows nothing about the cue can still finish most episodes by collecting objects
until the reward turns positive. Its *first* choice, however, is right one time
in three whatever it does afterwards.

Usage::

    python -m experiments.agency.run_reevaluate --grid-root runs/agency/grid \
        --output runs/agency/reevaluation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.agency.agent import GridAgent
from experiments.agency.baselines import GRUPolicy, LSTMPolicy
from experiments.agency.conditions import CONDITIONS, TASK
from experiments.agency.gridworld import (
    SymbolicForaging,
    latin_square_holdout,
    train_cues,
)
from experiments.agency.rollout import evaluate


def load_policy(condition: str, checkpoint: Path) -> GridAgent | GRUPolicy:
    agent_config = CONDITIONS[condition]
    if agent_config is None:
        control = LSTMPolicy if condition.startswith("lstm") else GRUPolicy
        policy = control(TASK, state_dim=64)
    else:
        policy = GridAgent(TASK, agent_config)
    policy.load_state_dict(torch.load(checkpoint, weights_only=True)["model_state_dict"])
    policy.eval()
    return policy


def reevaluate(grid_root: Path, *, repeats: int = 16, num_envs: int = 128) -> dict:
    """Evaluate every checkpoint on the train-cue and held-out-cue splits."""

    results: dict[str, dict[str, list[dict[str, float]]]] = {}
    for condition in CONDITIONS:
        for checkpoint in sorted((grid_root / condition).glob("seed*/checkpoint.pt")):
            seed = int(checkpoint.parent.name.removeprefix("seed"))
            torch.manual_seed(9_000 + seed)
            policy = load_policy(condition, checkpoint)
            entry = results.setdefault(condition, {"eval": [], "holdout": []})
            for split, cues, environment_seed in (
                ("eval", train_cues(TASK.num_colors, TASK.num_shapes), seed),
                (
                    "holdout",
                    latin_square_holdout(TASK.num_colors, TASK.num_shapes),
                    seed + 10_000,
                ),
            ):
                environment = SymbolicForaging(
                    TASK, num_envs, seed=environment_seed, allowed_cues=cues
                )
                entry[split].append(evaluate(environment, policy, repeats=repeats).as_dict())
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", type=Path, default=Path("runs/agency/grid"))
    parser.add_argument("--output", type=Path, default=Path("runs/agency/reevaluation.json"))
    parser.add_argument("--repeats", type=int, default=16)
    arguments = parser.parse_args()

    results = reevaluate(arguments.grid_root, repeats=arguments.repeats)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(results, indent=2, sort_keys=True))

    def mean(condition: str, split: str, metric: str) -> float:
        values = [entry[metric] for entry in results[condition][split]]
        return sum(values) / len(values)

    order = sorted(results, key=lambda name: -mean(name, "eval", "first_choice_accuracy"))
    print(f"{'condition':>22}  {'first-choice':>12} {'held-out':>9} {'return':>7} {'dist/ep':>8}")
    for condition in order:
        print(
            f"{condition:>22}  {mean(condition, 'eval', 'first_choice_accuracy'):12.3f} "
            f"{mean(condition, 'holdout', 'first_choice_accuracy'):9.3f} "
            f"{mean(condition, 'eval', 'mean_return'):7.3f} "
            f"{mean(condition, 'eval', 'distractor_rate'):8.2f}"
        )


if __name__ == "__main__":
    main()
