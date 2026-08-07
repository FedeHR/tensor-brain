"""A fixed offline corpus of Memory Maze trajectories.

This is what makes the filter study affordable. Rendering is the bottleneck --
roughly 200 environment steps per second, and it does not parallelise inside a
process -- so an on-policy method pays that cost again on every run. A filter
does not choose actions, so the trajectories can be rendered **once** and every
condition, seed and masking level replays the same bytes at GPU speed.

It also removes the confound that damaged the earlier studies in this line:
every architecture sees *byte-identical* data, so a difference between the
Tensor Brain and a recurrent control cannot be a difference in which states
their policies happened to visit.

Layout on disk, one directory per split::

    <root>/<split>/metadata.json
    <root>/<split>/shard_0000/image.npy        uint8   (episodes, steps, 64, 64, 3)
    <root>/<split>/shard_0000/action.npy       uint8   (episodes, steps)
    <root>/<split>/shard_0000/agent_pos.npy    float32 (episodes, steps, 2)
    ...

Separate ``.npy`` files per field rather than one ``.npz``, because ``.npy``
supports ``mmap_mode`` and ``.npz`` does not: a 10 GB corpus is then paged in by
the operating system on demand instead of being held in memory. ``in_memory``
overrides this on a node with the RAM to spare.

Images are stored as ``uint8``. Casting to ``float32`` at rest would quadruple
the corpus for no benefit -- the conversion belongs on the GPU at batch time.

Sizing: one 64x64x3 frame is 12,288 bytes, so a 1000-step episode is 12.3 MB and
the default 800-episode corpus is about 9.8 GB.

What is deliberately *not* stored: whether a target was visible. Visibility
depends on a field-of-view and an occlusion rule that are approximations, and
baking an approximation into the corpus would mean re-rendering to revise it.
The raw geometry (``targets_vec``, ``maze_layout``) is stored instead, and
visibility is derived at analysis time in ``horizon.py``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from experiments.agency.memorymaze.env import VectorMemoryMaze
from experiments.agency.memorymaze.explorer import ExplorerConfig, ScriptedExplorer

# Field name -> stored dtype. Everything the filter consumes or the probe scores
# against, and nothing derived: derived quantities can be recomputed, a missing
# recording cannot.
FIELDS: dict[str, str] = {
    "image": "uint8",
    "action": "uint8",
    "reward": "float32",
    "agent_pos": "float32",
    "agent_dir": "float32",
    "targets_pos": "float32",
    "targets_vec": "float32",
    "target_vec": "float32",
    "target_slot": "uint8",
}
# Stored once per episode rather than once per step, because it never changes
# within an episode.
EPISODE_FIELDS: dict[str, str] = {"maze_layout": "uint8"}


@dataclass(frozen=True)
class CorpusConfig:
    """Every controlled variable of the recording."""

    level: str = "9x9"
    # 600 / 100 / 100 at 1000 steps is about 9.8 GB and roughly 800k frames.
    # The filter has well under a million parameters and the probes are
    # 129-feature ridge regressions, so this is generous rather than tight.
    episodes: int = 600
    # The 9x9 level runs 250 simulated seconds at 4 Hz control, so an episode is
    # exactly 1000 steps. Truncating would shorten the only axis the
    # memory-horizon curve varies over.
    episode_steps: int = 1000
    # Both this and `episodes` must be multiples of `num_envs`, since episodes
    # are produced a full batch at a time. 24 at 1000 steps is a 295 MB shard.
    shard_episodes: int = 24
    # Environments stepped per process. Rendering does not parallelise inside a
    # process, so this buys convenience, not speed; speed comes from running
    # several record jobs over disjoint episode ranges.
    num_envs: int = 8
    seed: int = 0
    explorer: ExplorerConfig = field(default_factory=ExplorerConfig)

    def __post_init__(self) -> None:
        if self.episodes % self.num_envs:
            raise ValueError(
                f"episodes ({self.episodes}) must be a multiple of "
                f"num_envs ({self.num_envs}): episodes are recorded a batch at a time"
            )
        if self.shard_episodes % self.num_envs:
            raise ValueError(
                f"shard_episodes ({self.shard_episodes}) must be a multiple of "
                f"num_envs ({self.num_envs})"
            )
        if self.episodes % self.shard_episodes:
            raise ValueError(
                f"episodes ({self.episodes}) must be a multiple of "
                f"shard_episodes ({self.shard_episodes}), or the last shard is short"
            )


def _stack(observations: list[dict], key: str, dtype: str) -> np.ndarray:
    return np.stack([item[key] for item in observations]).astype(dtype)


def record_split(
    root: Path,
    split: str,
    config: CorpusConfig,
    *,
    progress: int = 5,
    shard_offset: int = 0,
) -> dict:
    """Render one split of the corpus and write it to ``root/split``.

    Environments are reused across episode groups. Memory Maze resamples the
    maze layout, the target positions and the spawn on every reset, so reuse
    costs no diversity and avoids recompiling MuJoCo models.

    ``shard_offset`` lets several jobs write disjoint parts of one split, which
    is how rendering is parallelised: rendering does not speed up inside a
    process, so throughput comes from running more of them. Each job must also
    be given a distinct ``seed``, or the "parallel" jobs would render the same
    mazes several times over.
    """

    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    environment = VectorMemoryMaze(
        config.num_envs, level=config.level, seed=config.seed, max_steps=config.episode_steps
    )
    explorer = ScriptedExplorer(config.num_envs, config.explorer, seed=config.seed)

    shard_index, episodes_done = shard_offset, 0
    pending: dict[str, list[np.ndarray]] = {}
    groups_per_shard = config.shard_episodes // config.num_envs

    for group in range(config.episodes // config.num_envs):
        explorer.reset()
        steps: dict[str, list[np.ndarray]] = {name: [] for name in FIELDS}
        # The layout is constant within an episode, so it is read once, up front.
        layout = _stack(environment.raw_observations(), "maze_layout", "uint8")

        for _ in range(config.episode_steps):
            observations = environment.raw_observations()
            action = explorer.act()
            steps["image"].append(_stack(observations, "image", "uint8"))
            for name in ("agent_pos", "agent_dir", "targets_pos", "targets_vec", "target_vec"):
                steps[name].append(_stack(observations, name, "float32"))
            steps["target_slot"].append(environment.target_slots().astype("uint8"))
            steps["action"].append(action.numpy().astype("uint8"))
            result = environment.step(action)
            steps["reward"].append(result.reward.numpy().astype("float32"))

        # (steps, envs, ...) -> (envs, steps, ...): the episode is the unit.
        for name in FIELDS:
            array = np.stack(steps[name], axis=1)
            pending.setdefault(name, []).append(array)
        pending.setdefault("maze_layout", []).append(layout)
        episodes_done += config.num_envs

        if (group + 1) % groups_per_shard == 0:
            _write_shard(directory, shard_index, pending)
            pending, shard_index = {}, shard_index + 1
            if progress and (shard_index - shard_offset) % progress == 0:
                print(
                    f"{split}: {episodes_done}/{config.episodes} episodes "
                    f"({shard_index} shards)",
                    flush=True,
                )

    if pending:
        _write_shard(directory, shard_index, pending)
        shard_index += 1

    environment.close()
    metadata = {
        "split": split,
        "shards": shard_index - shard_offset,
        "shard_offset": shard_offset,
        "episodes": config.episodes,
        "episode_steps": config.episode_steps,
        "config": asdict(config),
        "fields": FIELDS,
        "episode_fields": EPISODE_FIELDS,
    }
    # Written by every job that contributes to this split, so the last writer
    # wins. Only `episode_steps` is read back, and it is identical across jobs;
    # the shard *count* is deliberately not trusted -- `OfflineCorpus` globs.
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def _write_shard(directory: Path, index: int, pending: dict[str, list[np.ndarray]]) -> None:
    shard = directory / f"shard_{index:04d}"
    shard.mkdir(exist_ok=True)
    for name, arrays in pending.items():
        np.save(shard / f"{name}.npy", np.concatenate(arrays, axis=0))


class OfflineCorpus:
    """Read access to one recorded split, by sequence window or whole episode."""

    def __init__(self, root: Path, split: str, *, in_memory: bool = False) -> None:
        self.directory = Path(root) / split
        metadata_path = self.directory / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"no corpus at {self.directory}; record it with "
                f"`python -m experiments.agency.memorymaze.record`"
            )
        self.metadata = json.loads(metadata_path.read_text())
        self.episode_steps = int(self.metadata["episode_steps"])
        mode = None if in_memory else "r"
        # Globbed rather than counted: a split may be written by several jobs
        # over disjoint shard offsets, and each writes its own metadata, so no
        # single count is authoritative. What is on disk is.
        self.shards: list[dict[str, np.ndarray]] = []
        for shard in sorted(self.directory.glob("shard_*")):
            self.shards.append(
                {
                    name: np.load(shard / f"{name}.npy", mmap_mode=mode)
                    for name in (*FIELDS, *EPISODE_FIELDS)
                }
            )
        if not self.shards:
            raise FileNotFoundError(f"no shards under {self.directory}")
        # Flat episode index -> (shard, row within shard).
        self.locations: list[tuple[int, int]] = [
            (shard_index, row)
            for shard_index, shard in enumerate(self.shards)
            for row in range(shard["image"].shape[0])
        ]

    def __len__(self) -> int:
        return len(self.locations)

    def episode(self, index: int, start: int = 0, steps: int | None = None) -> dict[str, Tensor]:
        """One window of one episode, as torch tensors.

        Images come back as ``float32`` in [0, 1] and flattened, which is the
        contract ``PixelEncoder`` consumes; everything else keeps its shape.
        """

        shard_index, row = self.locations[index]
        shard = self.shards[shard_index]
        stop = self.episode_steps if steps is None else start + steps
        batch: dict[str, Tensor] = {}
        for name in FIELDS:
            values = np.asarray(shard[name][row, start:stop])
            batch[name] = torch.from_numpy(values.copy())
        batch["image"] = batch["image"].float().div_(255.0).reshape(stop - start, -1)
        batch["action"] = batch["action"].long()
        batch["target_slot"] = batch["target_slot"].long()
        batch["maze_layout"] = torch.from_numpy(np.asarray(shard["maze_layout"][row]).copy())
        return batch

    def sample(
        self, batch_size: int, steps: int, generator: torch.Generator
    ) -> dict[str, Tensor]:
        """A batch of random windows, stacked as ``(steps, batch, ...)``.

        Time-major, because every filter in this study consumes a step at a time
        and time-major slicing avoids a transpose in the inner loop.
        """

        episodes = torch.randint(
            len(self), (batch_size,), generator=generator
        ).tolist()
        starts = torch.randint(
            0, max(1, self.episode_steps - steps + 1), (batch_size,), generator=generator
        ).tolist()
        windows = [
            self.episode(episode, start, steps)
            for episode, start in zip(episodes, starts, strict=True)
        ]
        stacked = {
            name: torch.stack([window[name] for window in windows], dim=1)
            for name in FIELDS
        }
        stacked["maze_layout"] = torch.stack([window["maze_layout"] for window in windows])
        return stacked

    def episodes(self, count: int, *, start: int = 0) -> Iterator[dict[str, Tensor]]:
        """Whole episodes in order, for probing and for qualitative figures."""

        for index in range(start, min(start + count, len(self))):
            yield self.episode(index)
