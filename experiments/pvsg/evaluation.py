"""Evaluation loops shared by PVSG object experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

import torch
from torch.nn import functional as F

from experiments.pvsg.models import ObjectOutputs, PerceptionOutputs
from experiments.pvsg.runtime import candidate_tensors, category_candidates, move_features
from experiments.pvsg.supervision import (
    IGNORE_INDEX,
    build_category_targets,
    build_object_targets,
    build_predicate_targets,
)
from tb import IndexVocabulary


@dataclass
class _AccuracyTotals:
    """Loss and accuracy counts at observation, class, identity, and video levels."""

    loss: float = 0.0
    ignored: int = 0
    by_class: dict[int, list[int]] = field(default_factory=dict)
    by_identity: dict[str, list[int]] = field(default_factory=dict)
    by_video: dict[tuple[str, str], list[int]] = field(default_factory=dict)

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        identities: tuple[str, ...],
        videos: tuple[tuple[str, str], ...],
    ) -> None:
        valid = targets != IGNORE_INDEX
        self.ignored += len(targets) - int(valid.sum())
        if not bool(valid.any()):
            return
        valid_logits = logits[valid]
        valid_targets = targets[valid]
        self.loss += float(F.cross_entropy(valid_logits, valid_targets, reduction="sum"))
        predictions = valid_logits.argmax(-1).cpu().tolist()
        target_values = valid_targets.cpu().tolist()
        valid_rows = valid.cpu().nonzero(as_tuple=False).flatten().tolist()
        for prediction, target, row in zip(
            predictions, target_values, valid_rows, strict=True
        ):
            correct = int(prediction == target)
            for counts, key in (
                (self.by_class, target),
                (self.by_identity, identities[row]),
                (self.by_video, videos[row]),
            ):
                value = counts.setdefault(key, [0, 0])
                value[0] += correct
                value[1] += 1

    @property
    def support(self) -> int:
        return sum(total for _correct, total in self.by_class.values())

    @staticmethod
    def macro(counts: Mapping[Any, list[int]]) -> float:
        return fmean(correct / total for correct, total in counts.values())

    @property
    def micro(self) -> float:
        return sum(correct for correct, _total in self.by_class.values()) / self.support


@dataclass
class _ConditionalCategoryTotals:
    """Category accuracy after partitioning rows by identity correctness."""

    by_class: dict[int, list[int]] = field(default_factory=dict)

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        selected: torch.Tensor,
    ) -> None:
        valid = (targets != IGNORE_INDEX) & selected
        predictions = logits[valid].argmax(dim=-1).cpu().tolist()
        target_values = targets[valid].cpu().tolist()
        for prediction, target in zip(predictions, target_values, strict=True):
            counts = self.by_class.setdefault(target, [0, 0])
            counts[0] += int(prediction == target)
            counts[1] += 1

    @property
    def support(self) -> int:
        return sum(total for _correct, total in self.by_class.values())

    @property
    def micro(self) -> float:
        return sum(correct for correct, _total in self.by_class.values()) / self.support

    @property
    def class_macro(self) -> float:
        return fmean(correct / total for correct, total in self.by_class.values())


@torch.inference_mode()
def evaluate_objects(
    forward_object: Callable[
        [Mapping[str, Any], Mapping[str, torch.Tensor]], ObjectOutputs
    ],
    batches: Iterable[Mapping[str, Any]],
    vocabulary: IndexVocabulary,
    *,
    device: torch.device,
    hierarchy: Mapping[str, Any] | None,
    identities: bool,
) -> dict[str, float | int]:
    """Aggregate identity and category metrics without retaining predictions."""

    candidates = candidate_tensors(vocabulary, device)
    categories = category_candidates(candidates)
    totals = {group: _AccuracyTotals() for group in categories}
    conditional = (
        {
            condition: {
                group: _ConditionalCategoryTotals() for group in categories
            }
            for condition in ("identity_correct", "identity_incorrect")
        }
        if identities
        else None
    )
    identity_totals = _AccuracyTotals()
    examples = 0
    for cpu_batch in batches:
        batch = move_features(cpu_batch, ("scene_features", "object_features"), device)
        outputs = forward_object(batch, candidates)
        batch_size = len(batch["scene_features"])
        examples += batch_size
        batch_identities = tuple(cpu_batch["identity"])
        batch_videos = tuple(zip(cpu_batch["source"], cpu_batch["video_id"], strict=True))
        if identities:
            targets = build_object_targets(cpu_batch, vocabulary, hierarchy=hierarchy).to(device)
            identity_totals.update(
                outputs["identity_logits"],
                targets.identity,
                batch_identities,
                batch_videos,
            )
            identity_correct = (
                outputs["identity_logits"].argmax(dim=-1) == targets.identity
            )
            category_targets = targets.categories
        else:
            identity_correct = None
            category_targets = {
                group: target.to(device)
                for group, target in build_category_targets(
                    cpu_batch,
                    vocabulary,
                    hierarchy=hierarchy,
                    allow_unknown=True,
                ).items()
            }

        for group, target in category_targets.items():
            totals[group].update(
                outputs["category_logits"][group],
                target,
                batch_identities,
                batch_videos,
            )
            if conditional is not None and identity_correct is not None:
                for condition, selected in (
                    ("identity_correct", identity_correct),
                    ("identity_incorrect", ~identity_correct),
                ):
                    conditional[condition][group].update(
                        outputs["category_logits"][group], target, selected
                    )

    if not examples:
        raise ValueError("evaluation requires at least one example")
    result: dict[str, float | int] = {"examples": examples}
    category_losses = []
    category_accuracies = {name: [] for name in ("micro", "class", "identity", "video")}
    for group, total in totals.items():
        result[f"support/category/{group}"] = total.support
        result[f"ignored/category/{group}"] = total.ignored
        if total.support:
            loss = total.loss / total.support
            accuracies = {
                "micro": total.micro,
                "class": total.macro(total.by_class),
                "identity": total.macro(total.by_identity),
                "video": total.macro(total.by_video),
            }
            result[f"loss/category/{group}"] = loss
            result[f"accuracy/category/{group}"] = accuracies["micro"]
            for name in ("class", "identity", "video"):
                result[f"accuracy/category_{name}_macro/{group}"] = accuracies[name]
            result[f"classes/category/{group}"] = len(total.by_class)
            result[f"identities/category/{group}"] = len(total.by_identity)
            result[f"videos/category/{group}"] = len(total.by_video)
            category_losses.append(loss)
            for name, accuracy in accuracies.items():
                category_accuracies[name].append(accuracy)
    if not category_losses:
        raise ValueError("evaluation contains no supported category targets")
    result["loss/category_total"] = sum(category_losses)
    aggregate_names = {
        "micro": "observation_micro",
        "class": "class_macro",
        "identity": "identity_macro",
        "video": "video_macro",
    }
    for name, accuracies in category_accuracies.items():
        result[f"accuracy/category_level_mean_{aggregate_names[name]}"] = fmean(
            accuracies
        )
    if identities:
        result["loss/identity"] = identity_totals.loss / identity_totals.support
        result["accuracy/identity"] = identity_totals.micro
        result["accuracy/identity_macro"] = identity_totals.macro(
            identity_totals.by_identity
        )
        result["accuracy/identity_video_macro"] = identity_totals.macro(
            identity_totals.by_video
        )
        result["identities/identity"] = len(identity_totals.by_identity)
        result["videos/identity"] = len(identity_totals.by_video)
        assert conditional is not None
        for condition, totals_by_group in conditional.items():
            conditional_micro = []
            conditional_class_macro = []
            for group, total in totals_by_group.items():
                result[f"support/category_given_{condition}/{group}"] = total.support
                if not total.support:
                    continue
                result[f"accuracy/category_given_{condition}/{group}"] = total.micro
                result[
                    f"accuracy/category_class_macro_given_{condition}/{group}"
                ] = total.class_macro
                conditional_micro.append(total.micro)
                conditional_class_macro.append(total.class_macro)
            if conditional_micro:
                result[
                    f"accuracy/category_level_mean_observation_micro_given_{condition}"
                ] = fmean(conditional_micro)
                result[
                    f"accuracy/category_level_mean_class_macro_given_{condition}"
                ] = fmean(conditional_class_macro)
    return result


def predicate_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    predicate_names: Sequence[str],
    subject_categories: Sequence[str],
    object_categories: Sequence[str],
    videos: Sequence[tuple[str, str]],
    *,
    seen_triples: set[tuple[str, str, str]],
) -> dict[str, float | int]:
    """Aggregate positive-pair ranking metrics from one set of predictions."""

    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("predicate logits and targets must be equally shaped matrices")
    examples, candidate_count = logits.shape
    if not examples or len(predicate_names) != candidate_count:
        raise ValueError("predicate evaluation requires examples and one name per candidate")
    if not all(
        len(values) == examples
        for values in (subject_categories, object_categories, videos)
    ):
        raise ValueError("predicate metadata must align with the prediction rows")
    targets = targets.bool()
    target_counts = targets.sum(dim=-1)
    if bool((target_counts == 0).any()):
        raise ValueError("every evaluated pair must have a supported predicate")

    log_probabilities = logits.log_softmax(dim=-1)
    probabilities = log_probabilities.exp()
    distributions = targets.float() / target_counts.unsqueeze(-1)
    cross_entropy = -(distributions * log_probabilities).sum(dim=-1).mean()
    target_entropy = -torch.xlogy(distributions, distributions).sum(dim=-1).mean()
    result: dict[str, float | int] = {
        "examples": examples,
        "predicate_candidates": candidate_count,
        "predicate_assignments": int(target_counts.sum()),
        "loss/predicate_cross_entropy": float(cross_entropy),
        "loss/predicate_target_entropy": float(target_entropy),
        "loss/predicate_kl": float(cross_entropy - target_entropy),
    }

    max_target_count = int(target_counts.max())
    target_count_winners = logits.topk(max_target_count, dim=-1).indices
    exact = []
    for row, count in enumerate(target_counts.tolist()):
        predicted = set(target_count_winners[row, :count].tolist())
        expected = set(targets[row].nonzero(as_tuple=False).flatten().tolist())
        exact.append(predicted == expected)
    result["accuracy/predicate_exact_at_target_count"] = sum(exact) / examples

    for requested_k in (1, 5, 10):
        k = min(requested_k, candidate_count)
        winners = logits.topk(k, dim=-1).indices
        predicted = torch.zeros_like(targets).scatter(1, winners, True)
        hits = predicted & targets
        hits_by_row = hits.sum(dim=-1)
        recall_by_row = hits_by_row / target_counts
        result[f"recall/predicate_example_macro@{requested_k}"] = float(
            recall_by_row.float().mean()
        )
        result[f"recall/predicate_assignment_micro@{requested_k}"] = float(
            hits.sum() / target_counts.sum()
        )
        result[f"precision/predicate@{requested_k}"] = float(
            (hits_by_row.float() / k).mean()
        )
        class_recalls = []
        for predicate, name in enumerate(predicate_names):
            support = int(targets[:, predicate].sum())
            if not support:
                continue
            recall = float(hits[:, predicate].sum() / support)
            result[f"recall/predicate@{requested_k}/{name}"] = recall
            if requested_k == 1:
                result[f"support/predicate/{name}"] = support
            class_recalls.append(recall)
        result[f"recall/predicate_class_macro@{requested_k}"] = fmean(class_recalls)

        by_video: dict[tuple[str, str], list[float]] = {}
        for video, recall in zip(videos, recall_by_row.tolist(), strict=True):
            by_video.setdefault(video, []).append(recall)
        result[f"recall/predicate_video_macro@{requested_k}"] = fmean(
            fmean(values) for values in by_video.values()
        )

        triple_totals = {"seen": [0, 0], "unseen": [0, 0]}
        for row, (subject, object_) in enumerate(
            zip(subject_categories, object_categories, strict=True)
        ):
            for predicate in targets[row].nonzero(as_tuple=False).flatten().tolist():
                split = (
                    "seen"
                    if (subject, predicate_names[predicate], object_) in seen_triples
                    else "unseen"
                )
                triple_totals[split][0] += int(predicted[row, predicate])
                triple_totals[split][1] += 1
        for split, (correct, support) in triple_totals.items():
            result[f"support/triple_{split}"] = support
            if support:
                result[f"recall/triple_{split}@{requested_k}"] = correct / support

    average_precisions = []
    for predicate, name in enumerate(predicate_names):
        truth = targets[:, predicate]
        support = int(truth.sum())
        if not support:
            continue
        # The task is trained and evaluated as a categorical distribution. Raw logits
        # are not comparable across model families because their per-example normalizers
        # differ; use the corresponding categorical probability for cross-example AP.
        order = probabilities[:, predicate].argsort(descending=True)
        ranked_truth = truth[order].float()
        precision = ranked_truth.cumsum(dim=0) / torch.arange(
            1, examples + 1, device=logits.device
        )
        average_precision = float((precision * ranked_truth).sum() / support)
        result[f"average_precision/predicate/{name}"] = average_precision
        average_precisions.append(average_precision)
    result["mean_average_precision/predicate"] = fmean(average_precisions)
    return result


@torch.inference_mode()
def evaluate_pairs(
    forward_pair: Callable[
        [Mapping[str, Any], Mapping[str, torch.Tensor]], PerceptionOutputs
    ],
    batches: Iterable[Mapping[str, Any]],
    vocabulary: IndexVocabulary,
    *,
    device: torch.device,
    hierarchy: Mapping[str, Any] | None,
    seen_triples: set[tuple[str, str, str]],
) -> dict[str, float | int]:
    """Evaluate predicates and unary semantics without requiring held-out identities."""

    candidates = candidate_tensors(vocabulary, device)
    categories = category_candidates(candidates)
    category_totals = {
        owner: {group: _AccuracyTotals() for group in categories}
        for owner in ("subject", "object")
    }
    logits = []
    targets = []
    subject_categories = []
    object_categories = []
    videos = []
    feature_keys = (
        "scene_features",
        "subject_features",
        "object_features",
        "union_features",
    )
    for cpu_batch in batches:
        batch = move_features(cpu_batch, feature_keys, device)
        outputs = forward_pair(batch, candidates)
        logits.append(outputs["predicate_logits"].detach().cpu())
        targets.append(
            build_predicate_targets(
                cpu_batch, vocabulary, allow_unknown=True
            )
        )
        subject_categories.extend(cpu_batch["subject_category"])
        object_categories.extend(cpu_batch["object_category"])
        batch_videos = tuple(
            zip(cpu_batch["source"], cpu_batch["video_id"], strict=True)
        )
        videos.extend(batch_videos)
        for owner in ("subject", "object"):
            owner_identities = tuple(cpu_batch[f"{owner}_identity"])
            owner_targets = build_category_targets(
                {
                    "identity": owner_identities,
                    "category": tuple(cpu_batch[f"{owner}_category"]),
                },
                vocabulary,
                hierarchy=hierarchy,
                allow_unknown=True,
            )
            for group, totals in category_totals[owner].items():
                totals.update(
                    outputs[f"{owner}_category_logits"][group],
                    owner_targets[group].to(device),
                    owner_identities,
                    batch_videos,
                )
    if not logits:
        raise ValueError("evaluation requires at least one pair")
    result = predicate_metrics(
        torch.cat(logits),
        torch.cat(targets),
        vocabulary.group_labels("predicate"),
        subject_categories,
        object_categories,
        videos,
        seen_triples=seen_triples,
    )
    for owner, totals_by_group in category_totals.items():
        for group, totals in totals_by_group.items():
            result[f"ignored/{owner}_category/{group}"] = totals.ignored
            result[f"support/{owner}_category/{group}"] = totals.support
            if not totals.support:
                continue
            result[f"loss/{owner}_category/{group}"] = totals.loss / totals.support
            result[f"accuracy/{owner}_category/{group}"] = totals.micro
            result[f"accuracy/{owner}_category_class_macro/{group}"] = totals.macro(
                totals.by_class
            )
            result[f"accuracy/{owner}_category_video_macro/{group}"] = totals.macro(
                totals.by_video
            )
    return result
