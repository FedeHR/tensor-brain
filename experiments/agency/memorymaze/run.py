"""Train one condition and seed of the Memory Maze study.

Usage::

    MUJOCO_GL=glfw PYTHONPATH=src:. python -m experiments.agency.memorymaze.run \
        --level 9x9 --condition tb-full --seed 0 \
        --output-root runs/agency/memorymaze

This writes the training log, the trained weights and a metadata record. The
probe -- which is the actual claim of this study -- reads the weights back in
``run_probe.py``; training and probing are separate so that a probe can be
re-specified without paying for training again.

The environment needs Python 3.12 and the legacy ``gym``; see
``experiments/agency/memorymaze/env.py`` for the full recipe.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.agency.agent import AgentConfig
from experiments.agency.memorymaze.agent import MemoryMazeAgent, RecurrentControl
from experiments.agency.memorymaze.conditions import CONDITIONS, LEVELS
from experiments.agency.memorymaze.env import VectorMemoryMaze
from experiments.agency.minigrid.ppo import PPOConfig, train


def build_policy(
    condition: str, agent_config: AgentConfig | None
) -> MemoryMazeAgent | RecurrentControl:
    """Construct the Tensor Brain agent, or a control policy for ``None``."""

    if agent_config is not None:
        return MemoryMazeAgent(agent_config)
    cell = "lstm" if condition.startswith("lstm") else "gru"
    return RecurrentControl(CONDITIONS["tb-full"], cell=cell)


def run_condition(
    level_name: str,
    condition: str,
    seed: int,
    output_root: Path,
    *,
    updates: int | None = None,
) -> dict:
    """Train one condition at one seed and write its artifacts."""

    if condition not in CONDITIONS:
        raise KeyError(f"unknown condition {condition!r}")
    level = LEVELS[level_name]
    directory = output_root / level_name / condition / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)

    # These runs are launched nine at a time on an 11-core machine, so each
    # process must stay single-threaded or they oversubscribe and all slow down.
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    environment = VectorMemoryMaze(
        level.num_envs, level=level.env_id, seed=seed, max_steps=level.max_steps
    )
    policy = build_policy(condition, CONDITIONS[condition])
    ppo_config = PPOConfig(
        updates=updates or level.updates,
        segment_steps=level.segment_steps,
    )

    started = time.time()
    log = train(environment, policy, ppo_config)
    elapsed = time.time() - started
    environment.close()

    parameters = sum(p.numel() for p in policy.parameters())
    record = {
        "level": level_name,
        "env_id": level.env_id,
        "condition": condition,
        "seed": seed,
        "parameters": parameters,
        "frames": ppo_config.updates * level.segment_steps * level.num_envs,
        "num_envs": level.num_envs,
        "seconds": elapsed,
        "ppo": asdict(ppo_config),
        "agent": asdict(CONDITIONS[condition]) if CONDITIONS[condition] else None,
        "log": log.as_dict(),
    }
    (directory / "log.json").write_text(json.dumps(record, indent=2))
    torch.save(policy.state_dict(), directory / "policy.pt")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="9x9", choices=sorted(LEVELS))
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("runs/agency/memorymaze"))
    parser.add_argument(
        "--updates", type=int, default=None, help="override the level's budget"
    )
    arguments = parser.parse_args()

    record = run_condition(
        arguments.level,
        arguments.condition,
        arguments.seed,
        arguments.output_root,
        updates=arguments.updates,
    )
    final = record["log"]["train_metrics"][-1] if record["log"]["train_metrics"] else {}
    print(
        f"{arguments.condition} seed{arguments.seed}: "
        f"{record['frames']} frames in {record['seconds']:.0f}s, "
        f"mean_return={final.get('mean_return', float('nan')):.2f}"
    )


if __name__ == "__main__":
    main()
