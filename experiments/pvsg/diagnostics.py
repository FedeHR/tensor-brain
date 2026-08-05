"""Scalar diagnostics for PVSG Tensor Brain scale and feedback behavior."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from experiments.pvsg.models import IntegralTB, ObjectOutputs, PDirect, PerceptionOutputs


def _summary(values: Tensor) -> dict[str, float]:
    values = values.detach().float()
    flat = values.flatten()
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "min": float(flat.min()),
        "q01": float(torch.quantile(flat, 0.01)),
        "q10": float(torch.quantile(flat, 0.10)),
        "q50": float(torch.quantile(flat, 0.50)),
        "q90": float(torch.quantile(flat, 0.90)),
        "q99": float(torch.quantile(flat, 0.99)),
        "max": float(flat.max()),
    }


def _vector_summary(values: Tensor) -> dict[str, float]:
    summary = _summary(values)
    norms = values.detach().float().norm(dim=-1)
    summary.update(
        {
            "component_rms": float(values.detach().float().square().mean().sqrt()),
            "l2_mean": float(norms.mean()),
            "l2_std": float(norms.std(unbiased=False)),
        }
    )
    return summary


def _norm_ratio(numerator: Tensor, denominator: Tensor) -> float:
    numerator_norm = numerator.detach().float().norm(dim=-1)
    denominator_norm = denominator.detach().float().norm(dim=-1).clamp_min(1e-12)
    return float((numerator_norm / denominator_norm).mean())


def state_rows(trace: Mapping[str, Mapping[str, Tensor]]) -> list[dict[str, Any]]:
    """Summarize every recorded pre-CBS state and its corresponding CBS."""

    rows: list[dict[str, Any]] = []
    for window, tensors in trace.items():
        for name, values in tensors.items():
            if name != "input_drive" and not name.startswith("q_"):
                continue
            rows.append(
                {
                    "kind": "state",
                    "window": window,
                    "tensor": name,
                    **_vector_summary(values),
                }
            )
            if name.startswith("q_"):
                gamma = torch.sigmoid(values.detach().float())
                rows.append(
                    {
                        "kind": "cbs",
                        "window": window,
                        "tensor": name,
                        **_summary(gamma),
                        "fraction_below_0.01": float((gamma < 0.01).float().mean()),
                        "fraction_above_0.99": float((gamma > 0.99).float().mean()),
                    }
                )
    return rows


def operation_rows(trace: Mapping[str, Mapping[str, Tensor]]) -> list[dict[str, Any]]:
    """Measure how much input, feedback, and evolution move q and the CBS."""

    rows: list[dict[str, Any]] = []
    operations = (
        ("input", "q_before_input", "q_after_input"),
        ("feedback", "q_after_input", "q_after_feedback"),
        ("category_feedback", "q_after_feedback", "q_after_category_feedback"),
        ("evolution", "q_after_category_feedback", "q_after_evolution"),
        ("evolution", "q_after_feedback", "q_after_evolution"),
        ("evolution", "q_after_input", "q_after_evolution"),
    )
    for window, tensors in trace.items():
        recorded: set[str] = set()
        for operation, before_name, after_name in operations:
            if operation in recorded or before_name not in tensors or after_name not in tensors:
                continue
            before = tensors[before_name]
            after = tensors[after_name]
            for space, delta in (
                ("q", after - before),
                ("cbs", torch.sigmoid(after) - torch.sigmoid(before)),
            ):
                rows.append(
                    {
                        "kind": "operation_delta",
                        "window": window,
                        "operation": operation,
                        "space": space,
                        **_vector_summary(delta),
                    }
                )
            recorded.add(operation)
    return rows


def raw_input_rows(batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Summarize cached feature norms before experiment normalization."""

    rows = []
    for key, values in batch.items():
        if not key.endswith("_raw_l2") or not isinstance(values, Tensor):
            continue
        rows.append(
            {
                "kind": "raw_input_norm",
                "source": key.removesuffix("_raw_l2"),
                **_summary(values),
            }
        )
    return rows


def _direction_cosine_mean(values: Tensor) -> float:
    r"""Mean pairwise cosine between the batch's feedback vectors.

    A value near one means the injected vector barely depends on the example, so the
    pathway transports a constant that a no-feedback model can absorb into ``A``.
    Norm dispersion (``l2_std``) cannot distinguish that case from an informative one.
    """

    flat = values.detach().float().flatten(end_dim=-2)
    if flat.shape[0] < 2:
        return float("nan")
    unit = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    gram = unit @ unit.T
    count = gram.shape[0]
    off_diagonal_sum = float(gram.sum() - gram.diagonal().sum())
    return off_diagonal_sum / (count * (count - 1))


def feedback_rows(trace: Mapping[str, Mapping[str, Tensor]]) -> list[dict[str, Any]]:
    """Summarize actual and counterfactual index feedback at each entity window."""

    groups = (
        (
            "identity",
            "identity_probabilities",
            ("expected_feedback", "winner_feedback", "applied_feedback"),
        ),
        (
            "category",
            "category_probabilities",
            (
                "expected_category_feedback",
                "winner_category_feedback",
                "applied_category_feedback",
            ),
        ),
    )
    rows: list[dict[str, Any]] = []
    for window, tensors in trace.items():
        for group, probability_name, feedback_names in groups:
            if probability_name not in tensors:
                continue
            probabilities = tensors[probability_name].detach().float()
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            candidate_count = probabilities.shape[-1]
            rows.append(
                {
                    "kind": "attention",
                    "window": window,
                    "group": group,
                    "candidate_count": candidate_count,
                    "entropy_mean": float(entropy.mean()),
                    "normalized_entropy_mean": (
                        float((entropy / math.log(candidate_count)).mean())
                        if candidate_count > 1
                        else 0.0
                    ),
                    "maximum_probability_mean": float(
                        probabilities.max(dim=-1).values.mean()
                    ),
                }
            )
            for name in feedback_names:
                values = tensors[name]
                rows.append(
                    {
                        "kind": "feedback",
                        "window": window,
                        "group": group,
                        "tensor": name,
                        **_vector_summary(values),
                        "direction_cosine_mean": _direction_cosine_mean(values),
                        "l2_over_input_drive": _norm_ratio(
                            values, tensors["input_drive"]
                        ),
                        "l2_over_pre_feedback_q": _norm_ratio(
                            values, tensors["q_after_input"]
                        ),
                    }
                )
    return rows


def readout_rows(
    model: PDirect | IntegralTB,
    outputs: PerceptionOutputs,
    candidates_by_group: Mapping[str, Tensor],
) -> list[dict[str, Any]]:
    """Measure constant and state-dependent score scales for every active readout."""

    trace = outputs.get("trace")
    if trace is None:
        raise ValueError("readout diagnostics require a traced model forward pass")
    q_by_group: dict[str, tuple[str, Tensor]] = {
        "identity/subject": ("subject", trace["subject"]["q_after_input"]),
        "identity/object": ("object", trace["object"]["q_after_input"]),
        "predicate": ("predicate", trace["predicate"]["q_after_input"]),
    }
    category_q_name = (
        "q_after_feedback" if "q_after_feedback" in trace["subject"] else "q_after_input"
    )
    for group in candidates_by_group:
        if not group.startswith("object_category/"):
            continue
        q_by_group[f"{group}/subject"] = ("subject", trace["subject"][category_q_name])
        q_by_group[f"{group}/object"] = ("object", trace["object"][category_q_name])

    return _readout_scale_rows(model, q_by_group, candidates_by_group)


def _readout_scale_rows(
    model: PDirect | IntegralTB,
    q_by_group: Mapping[str, tuple[str, Tensor]],
    candidates_by_group: Mapping[str, Tensor],
) -> list[dict[str, Any]]:
    brain = model.brain
    rows: list[dict[str, Any]] = []
    for readout, (window, q) in q_by_group.items():
        base_group = readout.removesuffix("/subject").removesuffix("/object")
        if base_group.startswith("identity"):
            base_group = "identity"
        candidates = candidates_by_group[base_group].to(brain.A.device)
        columns = brain.A[:, candidates]
        bias = brain.index_bias(candidates)
        neutral_scores = brain.index_scores(q.new_zeros(brain.state_dim), candidates)
        data_scores = brain.index_scores(q, candidates) - neutral_scores
        neutral_std = neutral_scores.detach().float().std(unbiased=False)
        data_std = data_scores.detach().float().std(unbiased=False)
        rows.append(
            {
                "kind": "readout",
                "window": window,
                "group": readout,
                "candidate_count": int(candidates.numel()),
                "score_mode": brain.score_mode,
                "A_column_norm_mean": float(columns.detach().float().norm(dim=0).mean()),
                "A_column_norm_std": float(
                    columns.detach().float().norm(dim=0).std(unbiased=False)
                ),
                "score_offset_mean": float(bias.detach().float().mean()),
                "score_offset_std": float(bias.detach().float().std(unbiased=False)),
                "neutral_score_mean": float(neutral_scores.detach().float().mean()),
                "neutral_score_std": float(neutral_std),
                "data_dependent_score_mean": float(data_scores.detach().float().mean()),
                "data_dependent_score_std": float(data_std),
                "neutral_over_data_std": float(
                    neutral_std / data_std.clamp_min(1e-12)
                ),
            }
        )
    return rows


def object_readout_rows(
    model: PDirect | IntegralTB,
    outputs: ObjectOutputs,
    candidates_by_group: Mapping[str, Tensor],
) -> list[dict[str, Any]]:
    """Measure score scales for a single-entity schedule."""

    trace = outputs.get("trace")
    if trace is None:
        raise ValueError("readout diagnostics require a traced model forward pass")
    category_q = trace["object"].get(
        "q_after_feedback", trace["object"]["q_after_input"]
    )
    q_by_group = {"identity": ("object", trace["object"]["q_after_input"])}
    q_by_group.update(
        {
            group: ("object", category_q)
            for group in candidates_by_group
            if group.startswith("object_category/")
        }
    )
    return _readout_scale_rows(model, q_by_group, candidates_by_group)


def gradient_rows(
    model: PDirect | IntegralTB,
    candidates_by_group: Mapping[str, Tensor],
) -> list[dict[str, Any]]:
    """Summarize parameter gradients, including shared-index slices by role."""

    rows = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        rows.append(
            {
                "kind": "gradient",
                "parameter": name,
                "l2": float(parameter.grad.detach().float().norm()),
                "component_rms": float(parameter.grad.detach().float().square().mean().sqrt()),
            }
        )

    brain = model.brain
    if brain.A.grad is None:
        return rows
    for group, candidates in candidates_by_group.items():
        candidates = candidates.to(brain.A.device)
        gradients = [("brain.A", brain.A.grad[:, candidates])]
        if brain.a0 is not None and brain.a0.grad is not None:
            gradients.append(("brain.a0", brain.a0.grad[candidates]))
        for parameter_name, gradient in gradients:
            rows.append(
                {
                    "kind": "gradient_group",
                    "parameter": parameter_name,
                    "group": group,
                    "l2": float(gradient.detach().float().norm()),
                    "component_rms": float(gradient.detach().float().square().mean().sqrt()),
                }
            )
    return rows


def scale_trace_rows(
    model: PDirect | IntegralTB,
    outputs: PerceptionOutputs,
    candidates_by_group: Mapping[str, Tensor],
    batch: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return all scalar scale diagnostics for one forward/backward checkpoint."""

    trace = outputs.get("trace")
    if trace is None:
        raise ValueError("scale diagnostics require return_trace=True")
    return [
        *raw_input_rows(batch or {}),
        *state_rows(trace),
        *operation_rows(trace),
        *feedback_rows(trace),
        *readout_rows(model, outputs, candidates_by_group),
        *gradient_rows(model, candidates_by_group),
    ]


def object_scale_trace_rows(
    model: PDirect | IntegralTB,
    outputs: ObjectOutputs,
    candidates_by_group: Mapping[str, Tensor],
    batch: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the same diagnostics for the scene-and-single-entity schedule."""

    trace = outputs.get("trace")
    if trace is None:
        raise ValueError("scale diagnostics require return_trace=True")
    return [
        *raw_input_rows(batch or {}),
        *state_rows(trace),
        *operation_rows(trace),
        *feedback_rows(trace),
        *object_readout_rows(model, outputs, candidates_by_group),
        *gradient_rows(model, candidates_by_group),
    ]
