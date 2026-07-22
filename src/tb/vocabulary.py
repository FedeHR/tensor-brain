"""Parameter-free metadata for global Tensor Brain indices."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import torch
from jaxtyping import Int
from torch import Tensor


def get_candidate_positions(
    candidate_indices: Int[Tensor, " candidates"] | Sequence[int],
    global_indices: Int[Tensor, "*batch"] | int,
) -> Int[Tensor, "*batch"]:
    """Return each global index's position in an ordered candidate list.

    Tensor Brain scores are compact: ``scores[..., position]`` is the score for
    ``candidate_indices[position]``. Loss targets must therefore be candidate
    positions, even though measurement outcomes and embedding lookups use global
    indices.
    """

    if isinstance(global_indices, Tensor):
        device = global_indices.device
    elif isinstance(candidate_indices, Tensor):
        device = candidate_indices.device
    else:
        device = None

    candidates = torch.as_tensor(candidate_indices, dtype=torch.long, device=device)
    targets = torch.as_tensor(global_indices, dtype=torch.long, device=device)
    if candidates.ndim != 1 or candidates.numel() == 0:
        raise ValueError("candidate_indices must be a non-empty one-dimensional sequence")

    # Sorting lets us find arbitrary, non-contiguous global indices without relying
    # on an offset. `sorted_positions` maps the search result back to the original
    # candidate order, which is exactly the column order of the compact score tensor.
    sorted_candidates, sorted_positions = candidates.sort()
    if bool((sorted_candidates < 0).any()):
        raise ValueError("candidate_indices must contain non-negative global indices")
    if candidates.numel() > 1 and bool(
        (sorted_candidates[1:] == sorted_candidates[:-1]).any()
    ):
        raise ValueError("candidate_indices must not contain duplicate global indices")

    insertion_positions = torch.searchsorted(sorted_candidates, targets)
    safe_insertions = insertion_positions.clamp_max(candidates.numel() - 1)
    target_is_candidate = (insertion_positions < candidates.numel()) & (
        sorted_candidates[safe_insertions] == targets
    )
    if bool((~target_is_candidate).any()):
        raise ValueError("global_indices contain an index outside the candidate set")

    # Cross-entropy consumes these local positions. Do not use them to index A;
    # A is always indexed by the original global indices in `candidate_indices`.
    return sorted_positions[safe_insertions]


class IndexVocabulary:
    """Map symbolic names and candidate groups to stable global indices.

    The vocabulary owns no embeddings or other learned parameters. Global index
    ``k`` refers to ``TensorBrain.A[:, k]`` and ``TensorBrain.a0[k]``.
    """

    def __init__(self, labels: Sequence[str], groups: Mapping[str, Sequence[int]]) -> None:
        normalized_labels = tuple(labels)
        if not normalized_labels:
            raise ValueError("labels must not be empty")
        if len(set(normalized_labels)) != len(normalized_labels):
            raise ValueError("labels must be unique")
        if any(not label for label in normalized_labels):
            raise ValueError("labels must be non-empty strings")

        normalized_groups: dict[str, tuple[int, ...]] = {}
        for name, values in groups.items():
            indices = tuple(int(index) for index in values)
            if not name or not indices:
                raise ValueError("group names and candidate lists must not be empty")
            if len(set(indices)) != len(indices):
                raise ValueError(f"group {name!r} contains duplicate indices")
            if any(index < 0 or index >= len(normalized_labels) for index in indices):
                raise IndexError(f"group {name!r} contains an out-of-range index")
            normalized_groups[name] = indices

        self._labels = normalized_labels
        self._label_to_index = MappingProxyType(
            {label: index for index, label in enumerate(normalized_labels)}
        )
        self._groups = MappingProxyType(normalized_groups)

    @classmethod
    def from_groups(cls, groups: Mapping[str, Sequence[str]]) -> "IndexVocabulary":
        """Build stable global indices from possibly overlapping named groups."""

        labels: list[str] = []
        label_to_index: dict[str, int] = {}
        group_indices: dict[str, tuple[int, ...]] = {}
        for group, group_labels in groups.items():
            indices = []
            for label in group_labels:
                if label not in label_to_index:
                    label_to_index[label] = len(labels)
                    labels.append(label)
                indices.append(label_to_index[label])
            group_indices[group] = tuple(indices)
        return cls(labels, group_indices)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "IndexVocabulary":
        return cls(values["labels"], values["groups"])

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(self._groups)

    def __len__(self) -> int:
        return len(self._labels)

    def index(self, label: str) -> int:
        return self._label_to_index[label]

    def label(self, index: int) -> str:
        return self._labels[index]

    def indices(
        self, group: str, *, device: torch.device | str | None = None
    ) -> Int[Tensor, " indices"]:
        return torch.tensor(self._groups[group], dtype=torch.long, device=device)

    def get_positions(
        self,
        group: str,
        global_indices: Int[Tensor, "*batch"] | int,
    ) -> Int[Tensor, "*batch"]:
        """Return global indices' local score positions within a named group."""

        return get_candidate_positions(self._groups[group], global_indices)

    def group_labels(self, group: str) -> tuple[str, ...]:
        return tuple(self._labels[index] for index in self._groups[group])

    def to_dict(self) -> dict[str, object]:
        return {
            "labels": list(self._labels),
            "groups": {name: list(indices) for name, indices in self._groups.items()},
        }
