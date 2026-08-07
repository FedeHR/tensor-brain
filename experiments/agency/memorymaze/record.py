"""Render the offline Memory Maze corpus. Run once; every filter run replays it.

Usage::

    MUJOCO_GL=egl PYTHONPATH=src:. python -m experiments.agency.memorymaze.record \
        --root $MEMORYMAZE_CORPUS --split train --episodes 600

Rendering is the expensive half of this study and it does not parallelise inside
a process, so scale comes from running several of these over disjoint episode
ranges and concatenating the shards. ``--shard-offset`` exists for exactly that:
two jobs writing ``--shard-offset 0`` and ``--shard-offset 15`` into the same
split directory produce one corpus with no collisions.

Sizing, at 64x64x3 uint8 and 1000 steps per episode: 12.3 MB per episode, so the
default 600/100/100 split is about 9.8 GB.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.agency.memorymaze.corpus import CorpusConfig, record_split
from experiments.agency.memorymaze.explorer import ExplorerConfig

# Episodes per split in the reference corpus. Train is what Phase 1 fits on;
# the probe fits on `probe` and scores on `test`, and both are held out from
# Phase 1 so a probe never reads a state whose weights saw that episode.
# Multiples of both the batch width (8 environments) and the shard size (24),
# which `CorpusConfig` enforces.
DEFAULT_EPISODES = {"train": 576, "probe": 96, "test": 96}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", required=True, choices=sorted(DEFAULT_EPISODES))
    parser.add_argument("--level", default="9x9")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--episode-steps", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--shard-episodes", type=int, default=24)
    parser.add_argument("--dwell-mean", type=float, default=6.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="defaults to a per-split offset so splits never share a maze",
    )
    parser.add_argument(
        "--shard-offset",
        type=int,
        default=0,
        help="first shard index to write; use with --seed to split a job",
    )
    arguments = parser.parse_args()

    # Splits must not share mazes. A distinct seed base per split is what keeps
    # the probe's held-out score a statement about generalisation.
    seed_base = {"train": 0, "probe": 500_000, "test": 900_000}[arguments.split]
    config = CorpusConfig(
        level=arguments.level,
        episodes=arguments.episodes or DEFAULT_EPISODES[arguments.split],
        episode_steps=arguments.episode_steps,
        shard_episodes=arguments.shard_episodes,
        num_envs=arguments.num_envs,
        seed=seed_base if arguments.seed is None else arguments.seed,
        explorer=ExplorerConfig(dwell_mean=arguments.dwell_mean),
    )

    started = time.time()
    print(
        f"recording {arguments.split}: {config.episodes} episodes x "
        f"{config.episode_steps} steps "
        f"(~{config.episodes * config.episode_steps * 12288 / 1e9:.1f} GB)",
        flush=True,
    )
    metadata = record_split(
        arguments.root, arguments.split, config, shard_offset=arguments.shard_offset
    )
    elapsed = time.time() - started
    frames = config.episodes * config.episode_steps
    print(
        f"done: {metadata['shards']} shards, {frames} frames in {elapsed:.0f}s "
        f"({frames / max(elapsed, 1e-9):.0f} frames/s)"
    )
    print(json.dumps({"root": str(arguments.root), **metadata["config"]}, indent=2)[:400])


if __name__ == "__main__":
    main()
