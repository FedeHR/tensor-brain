"""Trainable and count-based controls for the PVSG comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from experiments.pvsg.models import (
    CategoryCandidates,
    ObjectOutputs,
    PerceptionOutputs,
)


def _readout(
    layer: nn.Linear,
    state: Float[Tensor, "*batch state"],
    candidates: Int[Tensor, " indices"],
) -> Float[Tensor, "*batch indices"]:
    return layer(state)[..., candidates]


class LinearProbe(nn.Module):
    """Conventional linear readouts on frozen local DINO evidence."""

    def __init__(self, state_dim: int, num_indices: int) -> None:
        super().__init__()
        self.readout = nn.Linear(state_dim, num_indices)

    def forward(
        self,
        subject_features: Float[Tensor, "*batch state"],
        object_features: Float[Tensor, "*batch state"],
        union_features: Float[Tensor, "*batch state"],
        identity_candidates: Int[Tensor, " identities"],
        predicate_candidates: Int[Tensor, " predicates"],
        *,
        category_candidates: CategoryCandidates | None = None,
    ) -> PerceptionOutputs:
        categories = category_candidates or {}
        return {
            "subject_identity_logits": _readout(
                self.readout, subject_features, identity_candidates
            ),
            "object_identity_logits": _readout(
                self.readout, object_features, identity_candidates
            ),
            "subject_category_logits": {
                group: _readout(self.readout, subject_features, candidates)
                for group, candidates in categories.items()
            },
            "object_category_logits": {
                group: _readout(self.readout, object_features, candidates)
                for group, candidates in categories.items()
            },
            "predicate_logits": _readout(
                self.readout, union_features, predicate_candidates
            ),
        }

    def forward_object(
        self,
        object_features: Float[Tensor, "*batch state"],
        identity_candidates: Int[Tensor, " identities"],
        *,
        category_candidates: CategoryCandidates | None = None,
    ) -> ObjectOutputs:
        return {
            "identity_logits": _readout(
                self.readout, object_features, identity_candidates
            ),
            "category_logits": {
                group: _readout(self.readout, object_features, candidates)
                for group, candidates in (category_candidates or {}).items()
            },
        }


class FusedLinear(nn.Module):
    """Linear readouts on the concatenation of all supplied evidence."""

    def __init__(self, state_dim: int, num_indices: int, *, num_sources: int) -> None:
        super().__init__()
        self.num_sources = num_sources
        self.readout = nn.Linear(num_sources * state_dim, num_indices)

    def _fuse(self, *features: Tensor) -> Tensor:
        if len(features) != self.num_sources:
            raise ValueError(
                f"expected {self.num_sources} feature sources, received {len(features)}"
            )
        return torch.cat(features, dim=-1)

    def forward_object(
        self,
        scene_features: Float[Tensor, "*batch state"],
        object_features: Float[Tensor, "*batch state"],
        identity_candidates: Int[Tensor, " identities"],
        *,
        category_candidates: CategoryCandidates | None = None,
    ) -> ObjectOutputs:
        fused = self._fuse(scene_features, object_features)
        return {
            "identity_logits": _readout(self.readout, fused, identity_candidates),
            "category_logits": {
                group: _readout(self.readout, fused, candidates)
                for group, candidates in (category_candidates or {}).items()
            },
        }

    def forward(
        self,
        scene_features: Float[Tensor, "*batch state"],
        subject_features: Float[Tensor, "*batch state"],
        object_features: Float[Tensor, "*batch state"],
        union_features: Float[Tensor, "*batch state"],
        identity_candidates: Int[Tensor, " identities"],
        predicate_candidates: Int[Tensor, " predicates"],
        *,
        category_candidates: CategoryCandidates | None = None,
    ) -> PerceptionOutputs:
        subject_state = self._fuse(
            scene_features, subject_features, object_features, union_features
        )
        object_state = self._fuse(
            scene_features, object_features, subject_features, union_features
        )
        categories = category_candidates or {}
        return {
            "subject_identity_logits": _readout(
                self.readout, subject_state, identity_candidates
            ),
            "object_identity_logits": _readout(
                self.readout, object_state, identity_candidates
            ),
            "subject_category_logits": {
                group: _readout(self.readout, subject_state, candidates)
                for group, candidates in categories.items()
            },
            "object_category_logits": {
                group: _readout(self.readout, object_state, candidates)
                for group, candidates in categories.items()
            },
            "predicate_logits": _readout(
                self.readout, subject_state, predicate_candidates
            ),
        }


class FlatFusion(nn.Module):
    """Non-TB MLP control receiving all evidence in one flat computation."""

    def __init__(self, state_dim: int, num_indices: int, hidden_dim: int) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(4 * state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.readout = nn.Linear(state_dim, num_indices)

    def _fuse(self, *features: Tensor) -> Tensor:
        return self.fusion(torch.cat(features, dim=-1))

    def forward(
        self,
        scene_features: Float[Tensor, "*batch state"],
        subject_features: Float[Tensor, "*batch state"],
        object_features: Float[Tensor, "*batch state"],
        union_features: Float[Tensor, "*batch state"],
        identity_candidates: Int[Tensor, " identities"],
        predicate_candidates: Int[Tensor, " predicates"],
        *,
        category_candidates: CategoryCandidates | None = None,
    ) -> PerceptionOutputs:
        subject_state = self._fuse(
            scene_features, subject_features, object_features, union_features
        )
        object_state = self._fuse(
            scene_features, object_features, subject_features, union_features
        )
        categories = category_candidates or {}
        return {
            "subject_identity_logits": _readout(
                self.readout, subject_state, identity_candidates
            ),
            "object_identity_logits": _readout(
                self.readout, object_state, identity_candidates
            ),
            "subject_category_logits": {
                group: _readout(self.readout, subject_state, candidates)
                for group, candidates in categories.items()
            },
            "object_category_logits": {
                group: _readout(self.readout, object_state, candidates)
                for group, candidates in categories.items()
            },
            "predicate_logits": _readout(
                self.readout, subject_state, predicate_candidates
            ),
        }


@dataclass(frozen=True)
class PredicatePriors:
    """Frequency and directed category-pair categorical priors."""

    frequency_logits: Float[Tensor, " predicates"]
    category_pair_logits: dict[tuple[str, str], Float[Tensor, " predicates"]]

    @classmethod
    def fit(
        cls,
        targets: Float[Tensor, "batch predicates"],
        subject_categories: Sequence[str],
        object_categories: Sequence[str],
        *,
        smoothing: float = 1.0,
    ) -> PredicatePriors:
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        if len(subject_categories) != len(targets) or len(object_categories) != len(targets):
            raise ValueError("categories and targets must contain the same number of rows")
        distributions = targets.float() / targets.sum(dim=-1, keepdim=True)

        def logits(rows: Tensor) -> Tensor:
            return (rows.sum(dim=0) + smoothing).log()

        grouped: dict[tuple[str, str], list[int]] = {}
        for row, pair in enumerate(zip(subject_categories, object_categories, strict=True)):
            grouped.setdefault(pair, []).append(row)
        return cls(
            frequency_logits=logits(distributions),
            category_pair_logits={
                pair: logits(distributions[rows]) for pair, rows in grouped.items()
            },
        )

    @classmethod
    def fit_records(
        cls,
        records: Sequence[Mapping[str, Any]],
        predicate_names: Sequence[str],
        *,
        smoothing: float = 1.0,
    ) -> PredicatePriors:
        """Fit the same priors directly from symbolic manifest rows."""

        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        positions = {name: position for position, name in enumerate(predicate_names)}
        frequency = torch.full((len(predicate_names),), smoothing)
        grouped: dict[tuple[str, str], Tensor] = {}
        for record in records:
            active = [positions[name] for name in record["predicates"] if name in positions]
            if not active:
                continue
            pair = (str(record["subject_category"]), str(record["object_category"]))
            counts = grouped.setdefault(pair, torch.full_like(frequency, smoothing))
            weight = 1.0 / len(active)
            frequency[active] += weight
            counts[active] += weight
        return cls(
            frequency_logits=frequency.log(),
            category_pair_logits={pair: counts.log() for pair, counts in grouped.items()},
        )

    def logits(
        self,
        subject_categories: Sequence[str],
        object_categories: Sequence[str],
    ) -> Float[Tensor, "batch predicates"]:
        """Return pair-conditioned logits, falling back to the frequency prior."""

        if len(subject_categories) != len(object_categories):
            raise ValueError("subject and object category batches must have the same length")
        return torch.stack(
            [
                self.category_pair_logits.get(pair, self.frequency_logits)
                for pair in zip(subject_categories, object_categories, strict=True)
            ]
        )
