"""Scalar diagnostics for PVSG Tensor Brain scale and feedback behavior."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from experiments.pvsg.models import IntegralTB, PDirect, PerceptionOutputs


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


def feedback_rows(trace: Mapping[str, Mapping[str, Tensor]]) -> list[dict[str, Any]]:
    """Summarize actual and counterfactual identity feedback at each entity window."""

    rows: list[dict[str, Any]] = []
    for window, tensors in trace.items():
        if "identity_probabilities" not in tensors:
            continue
        probabilities = tensors["identity_probabilities"].detach().float()
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        candidate_count = probabilities.shape[-1]
        rows.append(
            {
                "kind": "attention",
                "window": window,
                "candidate_count": candidate_count,
                "entropy_mean": float(entropy.mean()),
                "normalized_entropy_mean": (
                    float((entropy / math.log(candidate_count)).mean())
                    if candidate_count > 1
                    else 0.0
                ),
                "maximum_probability_mean": float(probabilities.max(dim=-1).values.mean()),
            }
        )
        for name in ("expected_feedback", "winner_feedback", "applied_feedback"):
            values = tensors[name]
            rows.append(
                {
                    "kind": "feedback",
                    "window": window,
                    "tensor": name,
                    **_vector_summary(values),
                    "l2_over_input_drive": _norm_ratio(values, tensors["input_drive"]),
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
    brain = model.brain
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

    rows: list[dict[str, Any]] = []
    for readout, (window, q) in q_by_group.items():
        base_group = readout.removesuffix("/subject").removesuffix("/object")
        if base_group.startswith("identity/"):
            base_group = "identity"
        candidates = candidates_by_group[base_group].to(brain.A.device)
        columns = brain.A[:, candidates]
        bias = brain.a0[candidates]
        neutral_scores = bias + 0.5 * columns.sum(dim=0)
        centered_scores = (torch.sigmoid(q) - 0.5) @ columns
        neutral_std = neutral_scores.detach().float().std(unbiased=False)
        centered_std = centered_scores.detach().float().std(unbiased=False)
        rows.append(
            {
                "kind": "readout",
                "window": window,
                "group": readout,
                "candidate_count": int(candidates.numel()),
                "A_column_norm_mean": float(columns.detach().float().norm(dim=0).mean()),
                "A_column_norm_std": float(
                    columns.detach().float().norm(dim=0).std(unbiased=False)
                ),
                "a0_mean": float(bias.detach().float().mean()),
                "a0_std": float(bias.detach().float().std(unbiased=False)),
                "neutral_score_mean": float(neutral_scores.detach().float().mean()),
                "neutral_score_std": float(neutral_std),
                "centered_score_mean": float(centered_scores.detach().float().mean()),
                "centered_score_std": float(centered_std),
                "neutral_over_centered_std": float(
                    neutral_std / centered_std.clamp_min(1e-12)
                ),
            }
        )
    return rows


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
    if brain.A.grad is None or brain.a0.grad is None:
        return rows
    for group, candidates in candidates_by_group.items():
        candidates = candidates.to(brain.A.device)
        for parameter_name, gradient in (
            ("brain.A", brain.A.grad[:, candidates]),
            ("brain.a0", brain.a0.grad[candidates]),
        ):
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
) -> list[dict[str, Any]]:
    """Return all scalar scale diagnostics for one forward/backward checkpoint."""

    trace = outputs.get("trace")
    if trace is None:
        raise ValueError("scale diagnostics require return_trace=True")
    return [
        *state_rows(trace),
        *feedback_rows(trace),
        *readout_rows(model, outputs, candidates_by_group),
        *gradient_rows(model, candidates_by_group),
    ]
