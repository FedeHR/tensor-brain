"""Evaluation loops shared by PVSG object experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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
    totals = {group: [0.0, 0, 0, 0] for group in categories}
    identity_loss = identity_correct = examples = 0
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
        if identities:
            targets = build_object_targets(cpu_batch, vocabulary, hierarchy=hierarchy).to(device)
            identity_loss += float(
                F.cross_entropy(outputs["identity_logits"], targets.identity, reduction="sum")
            )
            identity_correct += int(
                (outputs["identity_logits"].argmax(-1) == targets.identity).sum()
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
            valid = target != IGNORE_INDEX
            support = int(valid.sum())
            totals[group][2] += support
            totals[group][3] += batch_size - support
            if support:
                logits = outputs["category_logits"][group][valid]
                totals[group][0] += float(
                    F.cross_entropy(logits, target[valid], reduction="sum")
                )
                totals[group][1] += int((logits.argmax(-1) == target[valid]).sum())

    if not examples:
        raise ValueError("evaluation requires at least one example")
    result: dict[str, float | int] = {"examples": examples}
    category_losses = []
    category_accuracies = []
    for group, (loss, correct, support, ignored) in totals.items():
        result[f"support/category/{group}"] = support
        result[f"ignored/category/{group}"] = ignored
        if support:
            result[f"loss/category/{group}"] = loss / support
            result[f"accuracy/category/{group}"] = correct / support
            category_losses.append(loss / support)
            category_accuracies.append(correct / support)
    if not category_losses:
        raise ValueError("evaluation contains no supported category targets")
    result["loss/category_total"] = sum(category_losses)
    result["accuracy/category_macro"] = sum(category_accuracies) / len(
        category_accuracies
    )
    if identities:
        result["loss/identity"] = identity_loss / examples
        result["accuracy/identity"] = identity_correct / examples
    return result
