"""Evaluation loops shared by PVSG object experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Literal

import torch
from torch.nn import functional as F

from experiments.pvsg.models import IntegralTB
from experiments.pvsg.runtime import candidate_tensors, category_candidates, move_features
from experiments.pvsg.supervision import (
    IGNORE_INDEX,
    build_category_targets,
    build_object_targets,
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


@torch.inference_mode()
def evaluate_objects(
    model: IntegralTB,
    batches: Iterable[Mapping[str, Any]],
    vocabulary: IndexVocabulary,
    *,
    device: torch.device,
    hierarchy: Mapping[str, Any] | None,
    feedback_mode: Literal["p-sa", "p-samp"],
    identities: bool,
) -> dict[str, float | int]:
    """Aggregate identity and category metrics without retaining predictions."""

    candidates = candidate_tensors(vocabulary, device)
    categories = category_candidates(candidates)
    totals = {group: _AccuracyTotals() for group in categories}
    identity_totals = _AccuracyTotals()
    examples = 0
    model.eval()

    for cpu_batch in batches:
        batch = move_features(cpu_batch, ("scene_features", "object_features"), device)
        outputs = model.forward_object(
            batch["scene_features"],
            batch["object_features"],
            candidates["identity"],
            category_candidates=categories,
            feedback_mode=feedback_mode,
        )
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
            category_targets = targets.categories
        else:
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
    return result
