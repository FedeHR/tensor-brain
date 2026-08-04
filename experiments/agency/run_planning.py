"""Evaluate trained checkpoints under three action-selection readouts.

No retraining happens here. Each checkpoint was trained with generative
sampling; this script re-reads the same parameters under

* ``sample``  - the generative measurement the agent was trained with,
* ``argmax``  - winner-take-all decoding of the same distribution,
* ``planned`` - one-step imagination through the evolution operator, scored by
  the reward index (QTB Section 13.5.2).

Usage::

    python -m experiments.agency.run_planning --grid-root runs/agency/grid \
        --condition tb-full --output runs/agency/planning.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from experiments.agency.agent import GridAgent
from experiments.agency.conditions import CONDITIONS, TASK
from experiments.agency.gridworld import (
    SymbolicForaging,
    latin_square_holdout,
    train_cues,
)
from experiments.agency.rollout import evaluate

READOUTS = ("sample", "argmax", "planned")


def evaluate_readouts(
    grid_root: Path, condition: str, seeds: list[int], *, repeats: int = 16
) -> dict:
    """Evaluate one condition's checkpoints under every action-selection readout."""

    base = CONDITIONS[condition]
    if base is None:
        raise ValueError("planning readouts require a Tensor Brain agent")
    results: dict[str, dict[str, list[float]]] = {
        readout: {"eval": [], "holdout": []} for readout in READOUTS
    }
    for seed in seeds:
        checkpoint = grid_root / condition / f"seed{seed}" / "checkpoint.pt"
        if not checkpoint.exists():
            continue
        state = torch.load(checkpoint, weights_only=True)["model_state_dict"]
        for readout in READOUTS:
            torch.manual_seed(1234 + seed)
            agent = GridAgent(TASK, replace(base, action_selection=readout))
            agent.load_state_dict(state)
            environments = {
                "eval": SymbolicForaging(
                    TASK, 128, seed=seed, allowed_cues=train_cues(TASK.num_colors, TASK.num_shapes)
                ),
                "holdout": SymbolicForaging(
                    TASK,
                    128,
                    seed=seed + 10_000,
                    allowed_cues=latin_square_holdout(TASK.num_colors, TASK.num_shapes),
                ),
            }
            for split, environment in environments.items():
                results[readout][split].append(
                    evaluate(environment, agent, repeats=repeats).success_rate
                )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", type=Path, default=Path("runs/agency/grid"))
    parser.add_argument("--condition", default="tb-full")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("runs/agency/planning.json"))
    arguments = parser.parse_args()

    results = evaluate_readouts(arguments.grid_root, arguments.condition, arguments.seeds)
    payload = {"condition": arguments.condition, "seeds": arguments.seeds, "results": results}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2))
    for readout, splits in results.items():
        summary = " ".join(
            f"{split}={sum(values) / len(values):.3f}" for split, values in splits.items() if values
        )
        print(f"{readout:>8}: {summary}")


if __name__ == "__main__":
    main()
