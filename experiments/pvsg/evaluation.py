"""Evaluation loops shared by PVSG object experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

import torch
from torch.nn import functional as F

from experiments.pvsg.data import ELAPSED_PAIR_FIELDS
from experiments.pvsg.models import ObjectOutputs, PerceptionOutputs
from experiments.pvsg.runtime import candidate_tensors, category_candidates, move_features
from experiments.pvsg.supervision import (
    IGNORE_INDEX,
    build_category_targets,
    build_identity_targets,
    build_object_targets,
    build_predicate_targets,
)
from tb import IndexVocabulary

# Re-identification delay strata for the blocked protocol, in seconds since the
# participant was last observed. The paper's VRD-EX has no temporal axis at all.
DELAY_BIN_EDGES = (2.0, 5.0, 10.0)
DELAY_BIN_LABELS = ("0-2s", "2-5s", "5-10s", "10s+")


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
    detailed: bool = True,
) -> dict[str, float | int]:
    """Aggregate positive-pair ranking metrics from one set of predictions.

    ``detailed`` emits the per-predicate breakdown. Subsets evaluated only to compare
    strata against each other suppress it, because the aggregate is the comparison and
    the breakdown would multiply the result file by the number of strata.
    """

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
            if detailed:
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
        if detailed:
            result[f"average_precision/predicate/{name}"] = average_precision
        average_precisions.append(average_precision)
    result["mean_average_precision/predicate"] = fmean(average_precisions)
    return result


def batch_delays(batch: Mapping[str, Any]) -> torch.Tensor | None:
    """Return each pair's re-identification delay, or ``None`` outside blocked evaluation.

    The delay of a pair is that of its longer-unobserved participant, which is what
    governs how hard the pair is to recognize.
    """

    if not all(field in batch for field in ELAPSED_PAIR_FIELDS):
        return None
    subject_delay, object_delay = (
        batch[field].float() for field in ELAPSED_PAIR_FIELDS
    )
    return torch.maximum(subject_delay, object_delay)


def record_delays(records: Sequence[Mapping[str, Any]]) -> torch.Tensor | None:
    """Read delays straight from manifest rows, for models that never build a batch."""

    if not all(field in record for record in records for field in ELAPSED_PAIR_FIELDS):
        return None
    return torch.tensor(
        [
            max(float(record[field]) for field in ELAPSED_PAIR_FIELDS)
            for record in records
        ],
        dtype=torch.float32,
    )


def _delay_strata(delays: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """Bucket rows by how long the longer-unobserved participant had been absent."""

    boundaries = torch.tensor(DELAY_BIN_EDGES, dtype=delays.dtype)
    positions = torch.bucketize(delays, boundaries, right=True)
    return [
        (f"delay/{label}", positions == position)
        for position, label in enumerate(DELAY_BIN_LABELS)
    ]


def predicate_strata(
    logits: torch.Tensor,
    targets: torch.Tensor,
    predicate_names: Sequence[str],
    subject_categories: Sequence[str],
    object_categories: Sequence[str],
    videos: Sequence[tuple[str, str]],
    *,
    seen_triples: set[tuple[str, str, str]],
    strata: Sequence[tuple[str, torch.Tensor]],
) -> dict[str, float | int]:
    """Recompute the aggregate predicate metrics over each named subset of the rows."""

    result: dict[str, float | int] = {}
    for name, mask in strata:
        count = int(mask.sum())
        result[f"stratum/{name}/examples"] = count
        if not count:
            continue
        rows = mask.nonzero(as_tuple=False).flatten().tolist()
        subset = predicate_metrics(
            logits[mask],
            targets[mask],
            predicate_names,
            [subject_categories[row] for row in rows],
            [object_categories[row] for row in rows],
            [videos[row] for row in rows],
            seen_triples=seen_triples,
            detailed=False,
        )
        result.update({f"stratum/{name}/{key}": value for key, value in subset.items()})
    return result


def delay_metrics(
    delays: torch.Tensor | None,
    logits: torch.Tensor,
    targets: torch.Tensor,
    predicate_names: Sequence[str],
    subject_categories: Sequence[str],
    object_categories: Sequence[str],
    videos: Sequence[tuple[str, str]],
    *,
    seen_triples: set[tuple[str, str, str]],
) -> dict[str, float | int]:
    """Partition predicate quality by re-identification delay.

    Every model reports this on the blocked evaluation set, including the memoryless
    priors and the visual-only readout. Without a memoryless reference the delay axis is
    uninterpretable: predicate difficulty could itself correlate with elapsed time.
    """

    if delays is None:
        return {}
    result: dict[str, float | int] = {
        "delay_seconds_mean": float(delays.mean()),
        "delay_seconds_max": float(delays.max()),
    }
    result.update(
        predicate_strata(
            logits,
            targets,
            predicate_names,
            subject_categories,
            object_categories,
            videos,
            seen_triples=seen_triples,
            strata=_delay_strata(delays),
        )
    )
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
    identities: bool = False,
) -> dict[str, float | int]:
    """Evaluate predicates and unary semantics without requiring held-out identities.

    ``identities`` additionally scores the two identity readouts and partitions the
    predicate metrics by whether both participants were re-identified. It is meaningful
    only where the protocol enrolled the evaluated entities; on held-out video every
    candidate is wrong by construction. Where the manifest records a re-identification
    delay, the same partition is also reported per delay bucket.
    """

    candidates = candidate_tensors(vocabulary, device)
    categories = category_candidates(candidates)
    category_totals = {
        owner: {group: _AccuracyTotals() for group in categories}
        for owner in ("subject", "object")
    }
    identity_totals = {owner: _AccuracyTotals() for owner in ("subject", "object")}
    logits = []
    targets = []
    subject_categories = []
    object_categories = []
    videos = []
    pair_recognized = []
    pair_enrolled = []
    delays = []
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
        if identities:
            identity_targets = build_identity_targets(cpu_batch, vocabulary)
            recognized = torch.ones(len(batch_videos), dtype=torch.bool)
            enrolled = torch.ones(len(batch_videos), dtype=torch.bool)
            for owner, target in identity_targets.items():
                owner_logits = outputs[f"{owner}_identity_logits"]
                identity_totals[owner].update(
                    owner_logits,
                    target.to(device),
                    tuple(cpu_batch[f"{owner}_identity"]),
                    batch_videos,
                )
                valid = target != IGNORE_INDEX
                enrolled &= valid
                recognized &= valid & (owner_logits.argmax(dim=-1).cpu() == target)
            pair_recognized.append(recognized)
            pair_enrolled.append(enrolled)
        batch_delay = batch_delays(cpu_batch)
        if batch_delay is not None:
            delays.append(batch_delay)
    if not logits:
        raise ValueError("evaluation requires at least one pair")
    predicate_names = vocabulary.group_labels("predicate")
    all_logits = torch.cat(logits)
    all_targets = torch.cat(targets)
    result = predicate_metrics(
        all_logits,
        all_targets,
        predicate_names,
        subject_categories,
        object_categories,
        videos,
        seen_triples=seen_triples,
    )
    all_delays = torch.cat(delays) if delays else None
    result.update(
        delay_metrics(
            all_delays,
            all_logits,
            all_targets,
            predicate_names,
            subject_categories,
            object_categories,
            videos,
            seen_triples=seen_triples,
        )
    )
    if pair_recognized:
        recognized = torch.cat(pair_recognized)
        enrolled = torch.cat(pair_enrolled)
        result["support/identity_pair_enrolled"] = int(enrolled.sum())
        if bool(enrolled.any()):
            result["accuracy/identity_pair_exact"] = float(
                recognized[enrolled].float().mean()
            )
        result.update(
            predicate_strata(
                all_logits,
                all_targets,
                predicate_names,
                subject_categories,
                object_categories,
                videos,
                seen_triples=seen_triples,
                strata=[
                    ("identity_pair_correct", enrolled & recognized),
                    ("identity_pair_incorrect", enrolled & ~recognized),
                ],
            )
        )
        if all_delays is not None:
            # Recognition rate per delay bucket: does the memory itself decay?
            for name, mask in _delay_strata(all_delays):
                scored = mask & enrolled
                if bool(scored.any()):
                    result[f"stratum/{name}/accuracy/identity_pair_exact"] = float(
                        recognized[scored].float().mean()
                    )
    if identities:
        for owner, total in identity_totals.items():
            result[f"ignored/{owner}_identity"] = total.ignored
            result[f"support/{owner}_identity"] = total.support
            if not total.support:
                continue
            result[f"loss/{owner}_identity"] = total.loss / total.support
            result[f"accuracy/{owner}_identity"] = total.micro
            result[f"accuracy/{owner}_identity_macro"] = total.macro(total.by_identity)
            result[f"accuracy/{owner}_identity_video_macro"] = total.macro(
                total.by_video
            )
            result[f"identities/{owner}_identity"] = len(total.by_identity)
            result[f"videos/{owner}_identity"] = len(total.by_video)
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
