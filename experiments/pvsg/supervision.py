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
from experiments.pvsg.models import ObjectOutputs, PerceptionOutputs
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
class ObjectTargets:
    """Compact candidate positions for one object-observation batch."""

    identity: Int[Tensor, " batch"]
    categories: dict[str, Int[Tensor, " batch"]]

    def to(self, device: torch.device | str) -> ObjectTargets:
        return ObjectTargets(
            identity=self.identity.to(device),
            categories={group: target.to(device) for group, target in self.categories.items()},
        )


@dataclass(frozen=True)
class PairLosses:
    """Pair negative-log-likelihood terms with a mean category block."""

    total: Float[Tensor, ""]
    overfit_excess: Float[Tensor, ""]
    predicate_cross_entropy: Float[Tensor, ""]
    predicate_target_entropy: Float[Tensor, ""]
    predicate_kl: Float[Tensor, ""]
    subject_identity: Float[Tensor, ""]
    object_identity: Float[Tensor, ""]
    category_block: Float[Tensor, ""]
    subject_categories: dict[str, Float[Tensor, ""]]
    object_categories: dict[str, Float[Tensor, ""]]

    def scalars(self) -> dict[str, float]:
        values = {
            "loss/total": float(self.total.detach()),
            "loss/overfit_excess": float(self.overfit_excess.detach()),
            "loss/predicate_cross_entropy": float(
                self.predicate_cross_entropy.detach()
            ),
            "loss/predicate_target_entropy": float(
                self.predicate_target_entropy.detach()
            ),
            "loss/predicate_kl": float(self.predicate_kl.detach()),
            "loss/subject_identity": float(self.subject_identity.detach()),
            "loss/object_identity": float(self.object_identity.detach()),
            "loss/category_block": float(self.category_block.detach()),
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


@dataclass(frozen=True)
class ObjectLosses:
    """Unweighted identity and unary-category losses for an object batch."""

    total: Float[Tensor, ""]
    identity: Float[Tensor, ""]
    categories: dict[str, Float[Tensor, ""]]

    def scalars(self) -> dict[str, float]:
        return {
            "loss/total": float(self.total.detach()),
            "loss/identity": float(self.identity.detach()),
            **{
                f"loss/category/{group}": float(loss.detach())
                for group, loss in self.categories.items()
            },
        }


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
    labels: tuple[str | None, ...],
    vocabulary: IndexVocabulary,
    group: str,
    *,
    allow_unknown: bool = False,
) -> Int[Tensor, " batch"]:
    positions = []
    for label in labels:
        if label is None:
            positions.append(IGNORE_INDEX)
            continue
        try:
            positions.append(_candidate_position(vocabulary, group, label))
        except ValueError:
            if not allow_unknown:
                raise
            positions.append(IGNORE_INDEX)
    return torch.tensor(positions, dtype=torch.long)


def build_category_targets(
    batch: Mapping[str, Any],
    vocabulary: IndexVocabulary,
    *,
    hierarchy: Mapping[str, Any] | None = None,
    allow_unknown: bool = False,
) -> dict[str, Int[Tensor, " batch"]]:
    """Map object categories without requiring identities to be vocabulary members.

    Held-out-video identities are novel by construction, but their reviewed unary
    categories can still be evaluated. Unsupported category labels are ignored only
    when ``allow_unknown`` is explicitly enabled.
    """

    identities = _string_sequence(batch, "identity")
    source_categories = _string_sequence(batch, "category")
    if len(identities) != len(source_categories):
        raise ValueError("identity and category fields must have the same length")
    return {
        group: _class_targets(
            _category_labels(
                identities,
                source_categories,
                level=group.removeprefix(CATEGORY_GROUP_PREFIX),
                hierarchy=hierarchy,
            ),
            vocabulary,
            group,
            allow_unknown=allow_unknown,
        )
        for group in vocabulary.groups
        if group.startswith(CATEGORY_GROUP_PREFIX)
    }


def build_identity_targets(
    batch: Mapping[str, Any],
    vocabulary: IndexVocabulary,
    *,
    owners: Sequence[str] = ("subject", "object"),
) -> dict[str, Int[Tensor, " batch"]]:
    """Map pair participants to identity positions, ignoring unenrolled entities.

    Held-out-video entities are novel by construction and score ``IGNORE_INDEX`` at
    every candidate, so identity accuracy is only defined where the protocol enrolled
    the entity. The blocked protocol re-observes its training entities and therefore
    reports genuine identity support.
    """

    return {
        owner: _class_targets(
            _string_sequence(batch, f"{owner}_identity"),
            vocabulary,
            "identity",
            allow_unknown=True,
        )
        for owner in owners
    }


def _entity_targets(
    identities: tuple[str, ...],
    source_categories: tuple[str, ...],
    vocabulary: IndexVocabulary,
    hierarchy: Mapping[str, Any] | None,
) -> ObjectTargets:
    if len(identities) != len(source_categories):
        raise ValueError("identity and category fields must have the same length")
    global_indices = torch.tensor(
        [vocabulary.index(identity) for identity in identities], dtype=torch.long
    )
    categories = build_category_targets(
        {"identity": identities, "category": source_categories},
        vocabulary,
        hierarchy=hierarchy,
    )
    return ObjectTargets(
        identity=vocabulary.get_positions("identity", global_indices),
        categories=categories,
    )


def build_object_targets(
    batch: Mapping[str, Any],
    vocabulary: IndexVocabulary,
    *,
    hierarchy: Mapping[str, Any] | None = None,
) -> ObjectTargets:
    """Map one object-observation batch to identity and unary-category positions."""

    return _entity_targets(
        _string_sequence(batch, "identity"),
        _string_sequence(batch, "category"),
        vocabulary,
        hierarchy,
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

    subject_targets = _entity_targets(
        subject_identities, subject_categories, vocabulary, hierarchy
    )
    object_targets = _entity_targets(
        object_identities, object_categories, vocabulary, hierarchy
    )

    predicate_names = batch.get("predicates")
    if not isinstance(predicate_names, Sequence) or len(predicate_names) != batch_size:
        raise ValueError("batch['predicates'] must contain one label sequence per example")
    predicate_targets = build_predicate_targets(batch, vocabulary)

    return PairTargets(
        subject_identity=subject_targets.identity,
        object_identity=object_targets.identity,
        predicates=predicate_targets,
        subject_categories=subject_targets.categories,
        object_categories=object_targets.categories,
    )


def build_predicate_targets(
    batch: Mapping[str, Any],
    vocabulary: IndexVocabulary,
    *,
    allow_unknown: bool = False,
) -> Float[Tensor, "batch predicates"]:
    """Map predicate sets without requiring entity identities to be candidates."""

    predicate_names = batch["predicates"]
    if not isinstance(predicate_names, Sequence):
        raise ValueError("batch['predicates'] must contain one label sequence per example")
    batch_size = len(predicate_names)
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
            try:
                position = _candidate_position(vocabulary, "predicate", label)
            except ValueError:
                if allow_unknown:
                    continue
                raise
            predicate_targets[row, position] = 1.0
    if bool((predicate_targets.sum(dim=-1) == 0).any()):
        raise ValueError("every pair row must retain at least one supported predicate")
    return predicate_targets


def _masked_cross_entropy(
    logits: Float[Tensor, "batch categories"],
    target: Int[Tensor, " batch"],
) -> Float[Tensor, ""]:
    if bool((target != IGNORE_INDEX).any()):
        return F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
    return logits.sum() * 0.0


def pair_losses(outputs: PerceptionOutputs, targets: PairTargets) -> PairLosses:
    """Return pair losses with predicate/identity weights one and a mean category block."""

    predicate_distribution = targets.predicates / targets.predicates.sum(
        dim=-1, keepdim=True
    )
    predicate_log_probabilities = F.log_softmax(outputs["predicate_logits"], dim=-1)
    predicate_cross_entropy = -(
        predicate_distribution * predicate_log_probabilities
    ).sum(dim=-1).mean()
    predicate_target_entropy = -torch.xlogy(
        predicate_distribution, predicate_distribution
    ).sum(dim=-1).mean()
    predicate_kl = F.kl_div(
        predicate_log_probabilities,
        predicate_distribution,
        reduction="batchmean",
    )
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
    active_category_losses = [
        loss
        for owner_targets, owner_losses in (
            (targets.subject_categories, subject_categories),
            (targets.object_categories, object_categories),
        )
        for group, loss in owner_losses.items()
        if bool((owner_targets[group] != IGNORE_INDEX).any())
    ]
    category_block = (
        torch.stack(active_category_losses).mean()
        if active_category_losses
        else predicate_cross_entropy.new_zeros(())
    )
    total = predicate_cross_entropy + subject_identity + object_identity
    total = total + category_block
    return PairLosses(
        total=total,
        overfit_excess=total - predicate_target_entropy,
        predicate_cross_entropy=predicate_cross_entropy,
        predicate_target_entropy=predicate_target_entropy,
        predicate_kl=predicate_kl,
        subject_identity=subject_identity,
        object_identity=object_identity,
        category_block=category_block,
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
        predicate_targets = targets.predicates.bool()
        predicate_predictions = torch.zeros_like(predicate_targets)
        for row, count in enumerate(predicate_targets.sum(dim=-1).tolist()):
            winners = outputs["predicate_logits"][row].topk(count).indices
            predicate_predictions[row, winners] = True
        predicate_exact = (predicate_predictions == predicate_targets).all(dim=-1)
        predicate_recall = (
            (predicate_predictions & predicate_targets).sum(dim=-1)
            / predicate_targets.sum(dim=-1)
        )
        values = {
            "accuracy/subject_identity": float(subject_correct.float().mean()),
            "accuracy/object_identity": float(object_correct.float().mean()),
            "accuracy/predicate_exact": float(predicate_exact.float().mean()),
            "recall/predicate_at_target_count": float(predicate_recall.mean()),
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


def object_losses(outputs: ObjectOutputs, targets: ObjectTargets) -> ObjectLosses:
    """Return identity and enabled unary-category cross-entropies."""

    if set(outputs["category_logits"]) != set(targets.categories):
        raise ValueError("category output groups must exactly match category target groups")
    identity = F.cross_entropy(outputs["identity_logits"], targets.identity)
    categories = {
        group: _masked_cross_entropy(outputs["category_logits"][group], target)
        for group, target in targets.categories.items()
    }
    total = identity + sum(categories.values(), identity.new_zeros(()))
    return ObjectLosses(total=total, identity=identity, categories=categories)


def object_metrics(outputs: ObjectOutputs, targets: ObjectTargets) -> dict[str, float]:
    """Compute identity, category, and strict all-target object accuracy."""

    with torch.no_grad():
        all_correct = outputs["identity_logits"].argmax(dim=-1) == targets.identity
        values = {"accuracy/identity": float(all_correct.float().mean())}
        for group, target in targets.categories.items():
            valid = target != IGNORE_INDEX
            if not bool(valid.any()):
                continue
            correct = outputs["category_logits"][group].argmax(dim=-1)[valid] == target[valid]
            values[f"accuracy/category/{group}"] = float(correct.float().mean())
            complete = torch.ones_like(all_correct)
            complete[valid] = correct
            all_correct &= complete
        values["accuracy/all_exact"] = float(all_correct.float().mean())
        return values
