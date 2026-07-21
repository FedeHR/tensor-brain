"""Parameter-free metadata for global Tensor Brain indices."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import torch
from jaxtyping import Int
from torch import Tensor


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

    def group_labels(self, group: str) -> tuple[str, ...]:
        return tuple(self._labels[index] for index in self._groups[group])

    def to_dict(self) -> dict[str, object]:
        return {
            "labels": list(self._labels),
            "groups": {name: list(indices) for name, indices in self._groups.items()},
        }
