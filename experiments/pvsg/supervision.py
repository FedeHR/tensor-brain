"""Construct PVSG supervision tensors, losses, and overfit metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import functional as F

from experiments.pvsg.hierarchy import HIERARCHY_LEVELS, object_hierarchy_path
from experiments.pvsg.indices import predicate_label, source_category_label
from experiments.pvsg.models import PerceptionOutputs
from tb import IndexVocabulary

IGNORE_INDEX = -100
CATEGORY_GROUP_PREFIX = "object_category/"


@dataclass(frozen=True)
class PairTargets:
    """Compact candidate positions for one positive-pair batch."""

    subject_identity: Int[Tensor, " batch"]
    object_identity: Int[Tensor, " batch"]
    predicates: Float[Tensor, "batch predicates"]
    subject_categories: dict[str, Int[Tensor, " batch"]]
    object_categories: dict[str, Int[Tensor, " batch"]]

    def to(self, device: torch.device | str) -> PairTargets:
        return PairTargets(
            subject_identity=self.subject_identity.to(device),
            object_identity=self.object_identity.to(device),
            predicates=self.predicates.to(device),
            subject_categories={
                group: target.to(device) for group, target in self.subject_categories.items()
            },
            object_categories={
                group: target.to(device) for group, target in self.object_categories.items()
            },
        )


@dataclass(frozen=True)
class PairLosses:
    """Unweighted negative-log-likelihood terms for one pair batch."""

    total: Float[Tensor, ""]
    predicate_sum: Float[Tensor, ""]
    predicate_mean: Float[Tensor, ""]
    subject_identity: Float[Tensor, ""]
    object_identity: Float[Tensor, ""]
    subject_categories: dict[str, Float[Tensor, ""]]
    object_categories: dict[str, Float[Tensor, ""]]

    def scalars(self) -> dict[str, float]:
        values = {
            "loss/total": float(self.total.detach()),
            "loss/predicate_sum": float(self.predicate_sum.detach()),
            "loss/predicate_mean": float(self.predicate_mean.detach()),
            "loss/subject_identity": float(self.subject_identity.detach()),
            "loss/object_identity": float(self.object_identity.detach()),
        }
        for owner, losses in (
            ("subject", self.subject_categories),
            ("object", self.object_categories),
        ):
            values.update(
                {
                    f"loss/{owner}_category/{group}": float(loss.detach())
                    for group, loss in losses.items()
                }
            )
        return values


def _string_sequence(batch: Mapping[str, Any], key: str) -> tuple[str, ...]:
    values = batch[key]
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or any(not isinstance(value, str) for value in values)
    ):
        raise ValueError(f"batch[{key!r}] must be a sequence of strings")
    return tuple(values)


def _category_labels(
    identities: tuple[str, ...],
    source_categories: tuple[str, ...],
    *,
    level: str,
    hierarchy: Mapping[str, Any] | None,
) -> tuple[str | None, ...]:
    if level == "source":
        return tuple(source_category_label(category) for category in source_categories)
    if level not in HIERARCHY_LEVELS or hierarchy is None:
        raise ValueError(f"reviewed hierarchy is required for category level {level!r}")
    position = HIERARCHY_LEVELS.index(level)
    labels: list[str | None] = []
    for identity, source_category in zip(identities, source_categories, strict=True):
        try:
            path = object_hierarchy_path(
                hierarchy,
                identity_name=identity,
                source_category=source_category,
            )
        except KeyError:
            labels.append(None)
        else:
            labels.append(path[position])
    return tuple(labels)


def _candidate_position(
    vocabulary: IndexVocabulary,
    group: str,
    label: str,
) -> int:
    try:
        return int(vocabulary.get_positions(group, vocabulary.index(label)))
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"target {label!r} is outside the candidates for group {group!r}"
        ) from error


def _class_targets(
    labels: tuple[str | None, ...], vocabulary: IndexVocabulary, group: str
) -> Int[Tensor, " batch"]:
    return torch.tensor(
        [
            IGNORE_INDEX if label is None else _candidate_position(vocabulary, group, label)
            for label in labels
        ],
        dtype=torch.long,
    )


def build_pair_targets(
    batch: Mapping[str, Any],
    vocabulary: IndexVocabulary,
    *,
    hierarchy: Mapping[str, Any] | None = None,
) -> PairTargets:
    """Map symbolic pair labels to compact score positions and multi-hot predicates."""

    subject_identities = _string_sequence(batch, "subject_identity")
    object_identities = _string_sequence(batch, "object_identity")
    subject_categories = _string_sequence(batch, "subject_category")
    object_categories = _string_sequence(batch, "object_category")
    batch_size = len(subject_identities)
    if not all(
        len(values) == batch_size
        for values in (object_identities, subject_categories, object_categories)
    ):
        raise ValueError("pair batch symbolic fields must have the same length")

    subject_global = torch.tensor(
        [vocabulary.index(identity) for identity in subject_identities], dtype=torch.long
    )
    object_global = torch.tensor(
        [vocabulary.index(identity) for identity in object_identities], dtype=torch.long
    )
    subject_identity_targets = vocabulary.get_positions("identity", subject_global)
    object_identity_targets = vocabulary.get_positions("identity", object_global)

    predicate_names = batch["predicates"]
    if not isinstance(predicate_names, Sequence) or len(predicate_names) != batch_size:
        raise ValueError("batch['predicates'] must contain one label sequence per example")
    predicate_targets = torch.zeros(
        batch_size, len(vocabulary.group_labels("predicate")), dtype=torch.float32
    )
    for row, predicates in enumerate(predicate_names):
        if (
            not isinstance(predicates, Sequence)
            or isinstance(predicates, (str, bytes))
            or any(not isinstance(predicate, str) for predicate in predicates)
        ):
            raise ValueError("every predicate target must be a sequence of strings")
        for predicate in predicates:
            label = predicate_label(predicate)
            predicate_targets[
                row, _candidate_position(vocabulary, "predicate", label)
            ] = 1.0

    subject_category_targets = {}
    object_category_targets = {}
    for group in vocabulary.groups:
        if not group.startswith(CATEGORY_GROUP_PREFIX):
            continue
        level = group.removeprefix(CATEGORY_GROUP_PREFIX)
        subject_labels = _category_labels(
            subject_identities,
            subject_categories,
            level=level,
            hierarchy=hierarchy,
        )
        object_labels = _category_labels(
            object_identities,
            object_categories,
            level=level,
            hierarchy=hierarchy,
        )
        subject_category_targets[group] = _class_targets(
            subject_labels, vocabulary, group
        )
        object_category_targets[group] = _class_targets(
            object_labels, vocabulary, group
        )

    return PairTargets(
        subject_identity=subject_identity_targets,
        object_identity=object_identity_targets,
        predicates=predicate_targets,
        subject_categories=subject_category_targets,
        object_categories=object_category_targets,
    )


def _masked_cross_entropy(
    logits: Float[Tensor, "batch categories"],
    target: Int[Tensor, " batch"],
) -> Float[Tensor, ""]:
    if bool((target != IGNORE_INDEX).any()):
        return F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
    return logits.sum() * 0.0


def pair_losses(outputs: PerceptionOutputs, targets: PairTargets) -> PairLosses:
    """Return the factorized pair negative log-likelihood without ad-hoc weights."""

    predicate_per_label = F.binary_cross_entropy_with_logits(
        outputs["predicate_logits"], targets.predicates, reduction="none"
    )
    predicate_sum = predicate_per_label.sum(dim=-1).mean()
    predicate_mean = predicate_per_label.mean()
    subject_identity = F.cross_entropy(
        outputs["subject_identity_logits"], targets.subject_identity
    )
    object_identity = F.cross_entropy(
        outputs["object_identity_logits"], targets.object_identity
    )
    if set(outputs["subject_category_logits"]) != set(targets.subject_categories) or set(
        outputs["object_category_logits"]
    ) != set(targets.object_categories):
        raise ValueError("category output groups must exactly match category target groups")
    subject_categories = {
        group: _masked_cross_entropy(outputs["subject_category_logits"][group], target)
        for group, target in targets.subject_categories.items()
    }
    object_categories = {
        group: _masked_cross_entropy(outputs["object_category_logits"][group], target)
        for group, target in targets.object_categories.items()
    }
    total = predicate_sum + subject_identity + object_identity
    total = total + sum(subject_categories.values(), total.new_zeros(()))
    total = total + sum(object_categories.values(), total.new_zeros(()))
    return PairLosses(
        total=total,
        predicate_sum=predicate_sum,
        predicate_mean=predicate_mean,
        subject_identity=subject_identity,
        object_identity=object_identity,
        subject_categories=subject_categories,
        object_categories=object_categories,
    )


def pair_metrics(outputs: PerceptionOutputs, targets: PairTargets) -> dict[str, float]:
    """Compute strict finite-set metrics used to decide whether overfitting succeeded."""

    with torch.no_grad():
        subject_correct = (
            outputs["subject_identity_logits"].argmax(dim=-1)
            == targets.subject_identity
        )
        object_correct = (
            outputs["object_identity_logits"].argmax(dim=-1)
            == targets.object_identity
        )
        predicate_predictions = outputs["predicate_logits"] >= 0
        predicate_targets = targets.predicates.bool()
        predicate_exact = (predicate_predictions == predicate_targets).all(dim=-1)
        values = {
            "accuracy/subject_identity": float(subject_correct.float().mean()),
            "accuracy/object_identity": float(object_correct.float().mean()),
            "accuracy/predicate_exact": float(predicate_exact.float().mean()),
            "accuracy/predicate_labels": float(
                (predicate_predictions == predicate_targets).float().mean()
            ),
        }
        all_correct = subject_correct & object_correct & predicate_exact
        for owner, logits_by_group, targets_by_group in (
            ("subject", outputs["subject_category_logits"], targets.subject_categories),
            ("object", outputs["object_category_logits"], targets.object_categories),
        ):
            for group, target in targets_by_group.items():
                valid = target != IGNORE_INDEX
                if not bool(valid.any()):
                    continue
                correct = logits_by_group[group].argmax(dim=-1)[valid] == target[valid]
                values[f"accuracy/{owner}_category/{group}"] = float(
                    correct.float().mean()
                )
                owner_complete = torch.ones_like(all_correct)
                owner_complete[valid] = correct
                all_correct &= owner_complete
        values["accuracy/all_exact"] = float(all_correct.float().mean())
        return values
