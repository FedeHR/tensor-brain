"""Stage A: the capacity gate, on a fully observed variant of the task.

Reinforcement learning conflates three questions: can the architecture *express*
a cue-conditioned policy, can it *learn* one from reward, and can it *explore*.
This gate isolates the first. The view radius covers the whole grid, so the
privileged oracle's action is a deterministic function of the observation and
the cue, and a perfect model would reach agreement 1.0.

The informative contrast is ``tb-full`` against ``no-cue``: without the
instruction indices the target is unidentifiable among the distractors, so
agreement must fall to roughly the level of a policy that walks towards an
arbitrary object.

Usage::

    python -m experiments.agency.run_clone_gate --output runs/agency/clone_gate.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from experiments.agency.agent import GridAgent
from experiments.agency.conditions import REFERENCE, TASK
from experiments.agency.gridworld import (
    SymbolicForaging,
    latin_square_holdout,
    train_cues,
)
from experiments.agency.training import CloneConfig, clone_from_oracle, split_evaluator

# Fully observed: a radius of size-1 reaches every cell from every cell.
OBSERVED_TASK = replace(TASK, view_radius=TASK.size - 1)

GATE_CONDITIONS = {
    "tb-full": REFERENCE,
    "no-cue": replace(REFERENCE, cue_mode="none"),
    "no-percept-measure": replace(REFERENCE, measure_percepts=False),
    "deliberate-3-attend": replace(REFERENCE, deliberation_windows=3),
    "state-128": replace(REFERENCE, state_dim=128, hidden_dim=128),
}


def run_gate(
    updates: int, seeds: list[int], *, num_envs: int = 128, learning_rate: float = 1e-3
) -> dict:
    """Clone every gate condition and report the teacher-forced agreement."""

    results: dict[str, dict] = {}
    for name, agent_config in GATE_CONDITIONS.items():
        curves, finals, successes = [], [], []
        for seed in seeds:
            torch.manual_seed(seed)
            environment = SymbolicForaging(
                OBSERVED_TASK,
                num_envs,
                seed=seed,
                allowed_cues=train_cues(OBSERVED_TASK.num_colors, OBSERVED_TASK.num_shapes),
            )
            agent = GridAgent(OBSERVED_TASK, agent_config)
            holdout = SymbolicForaging(
                OBSERVED_TASK,
                num_envs,
                seed=seed + 10_000,
                allowed_cues=latin_square_holdout(
                    OBSERVED_TASK.num_colors, OBSERVED_TASK.num_shapes
                ),
            )
            log = clone_from_oracle(
                environment,
                agent,
                CloneConfig(updates=updates, learning_rate=learning_rate),
                evaluation=split_evaluator(agent, environment, holdout, repeats=2),
                evaluate_every=max(1, updates // 40),
            )
            curves.append(
                {
                    "episodes": log.episodes,
                    "agreement": [point["action_agreement"] for point in log.train_metrics],
                    "success": [point["success_rate"] for point in log.eval_metrics],
                    "holdout_success": [point["success_rate"] for point in log.holdout_metrics],
                    "loss": log.loss,
                }
            )
            finals.append(curves[-1]["agreement"][-1])
            successes.append(curves[-1]["success"][-1])
        results[name] = {
            "curves": curves,
            "final_agreement": finals,
            "final_success": successes,
        }
        print(
            f"{name:>22}: oracle agreement {sum(finals) / len(finals):.3f}  "
            f"rollout success {sum(successes) / len(successes):.3f}"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("runs/agency/clone_gate.json"))
    arguments = parser.parse_args()

    results = run_gate(arguments.updates, arguments.seeds)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            {"task": asdict(OBSERVED_TASK), "updates": arguments.updates, "results": results},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
