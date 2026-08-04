"""Probe a trained Memory Maze policy against ground truth.

Usage::

    uv run --frozen --no-sync --python 3.12 python -m \
        experiments.agency.memorymaze.run_probe \
        --run-root runs/agency/memorymaze/9x9 --output probe.json

Probing is separate from training so that a probe can be re-specified without
paying for training again, which matters here because training is the expensive
half and the probe is the half carrying the claim.

Every trained policy is probed against an **untrained policy of the same
architecture**. That control is not optional: the convolutional encoder is
shared and randomly initialised, and a random projection of pixels already
carries some position information. Without the control, a probe would be
reporting what the camera sees rather than what the agent retained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.agency.memorymaze.conditions import CONDITIONS, LEVELS
from experiments.agency.memorymaze.env import COLOR_NAMES, VectorMemoryMaze
from experiments.agency.memorymaze.probe import (
    REGRESSION_TARGETS,
    native_readout,
    probe_colors,
    probe_regression,
    record,
    write_test,
)
from experiments.agency.memorymaze.run import build_policy


def probe_policy(
    policy,
    level_name: str,
    *,
    steps: int,
    seed: int,
    warmup: int,
) -> dict:
    """Fit and score every probe for one policy."""

    level = LEVELS[level_name]
    # Two rollouts from different environment seeds. Adjacent steps within one
    # rollout are strongly correlated, so a split inside a single rollout would
    # leak and report an R^2 that means nothing.
    train_env = VectorMemoryMaze(
        level.num_envs, level=level.env_id, seed=seed, max_steps=level.max_steps
    )
    test_env = VectorMemoryMaze(
        level.num_envs, level=level.env_id, seed=seed + 1000, max_steps=level.max_steps
    )
    try:
        train = record(train_env, policy, steps, warmup=warmup)
        test = record(test_env, policy, steps, warmup=warmup)
        results = {
            "samples_train": len(train),
            "samples_test": len(test),
            "linear_probe": {
                key: probe_regression(train, test, key) for key in REGRESSION_TARGETS
            },
            "colour_probe": probe_colors(train, test),
            "native_readout": native_readout(test),
        }
    finally:
        train_env.close()
        test_env.close()

    write_env = VectorMemoryMaze(
        level.num_envs, level=level.env_id, seed=seed + 2000, max_steps=level.max_steps
    )
    try:
        results["write_test"] = {
            color: write_test(write_env, policy, color=color) for color in COLOR_NAMES
        }
    finally:
        write_env.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="directory holding <condition>/seed<n>/policy.pt",
    )
    parser.add_argument("--level", default="9x9", choices=sorted(LEVELS))
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    torch.set_num_threads(1)
    results: dict[str, dict] = {}
    for condition in sorted(CONDITIONS):
        for directory in sorted(arguments.run_root.glob(f"{condition}/seed*")):
            checkpoint = directory / "policy.pt"
            if not checkpoint.exists():
                continue
            seed = int(directory.name.removeprefix("seed"))
            torch.manual_seed(arguments.seed)

            trained = build_policy(condition, CONDITIONS[condition])
            trained.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            trained.eval()
            # The control: same architecture, same shared encoder, no training.
            untrained = build_policy(condition, CONDITIONS[condition])
            untrained.eval()

            key = f"{condition}/seed{seed}"
            results[key] = {
                "trained": probe_policy(
                    trained, arguments.level,
                    steps=arguments.steps, seed=arguments.seed, warmup=arguments.warmup,
                ),
                "untrained": probe_policy(
                    untrained, arguments.level,
                    steps=arguments.steps, seed=arguments.seed, warmup=arguments.warmup,
                ),
            }
            probe = results[key]["trained"]["linear_probe"]["targets_pos"]["r2"]
            control = results[key]["untrained"]["linear_probe"]["targets_pos"]["r2"]
            print(f"{key}: targets_pos R2 trained={probe:.3f} untrained={control:.3f}")

    if not results:
        raise SystemExit(f"no checkpoints found under {arguments.run_root}")
    output = arguments.output or arguments.run_root / "probe.json"
    output.write_text(json.dumps(results, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
