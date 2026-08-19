"""The tau instrument, run on three public rejection-sampling corpora.

The cancellation identity needs a corpus whose *recorded count* rises with the
score mass. Write the count rule as

    E[c_i] = exp(alpha + tau * log m_i)

with ``m_i`` a proxy for the query's score mass. Then

    tau = 1  the count tracks the score mass    -> gate; additive rule exact
    tau = 0  the count is fixed per query       -> conditioning on a total;
                                                   the normalizer is reinstated
    tau < 0  the count is anti-correlated       -> deliberate anti-gate

DART-Math (arXiv:2407.13690, NeurIPS'24) diagnoses vanilla rejection tuning as
"severely biased towards easy queries" *because* the retained count tracks the
pass rate, and ships two corpora built to remove that dependence
(DARS-Uniform: equal count per query; DARS-Prop2Diff: count rising with
difficulty). This measures where the three corpora actually sit.

WHAT THE DATA DOES AND DOES NOT SUPPORT
---------------------------------------
`dart-math-pool-math` retains only *accepted* responses (`ans_correct` is True
on every row). So the number of raw attempts per query is **not recoverable**
from the public pool, and the Poisson offset `log n_i` cannot be formed. Two
consequences, both stated rather than worked around:

  * the score-mass proxy is the pool's retained count `m_i`, which is
    proportional to the pass rate `p_i` only under the assumption that DART
    sampled a roughly constant raw budget per query. That assumption is the
    weakest link in this measurement.
  * the VRT row is therefore **definitional, not measured**: vanilla rejection
    tuning is "keep every accepted response", which is the pool itself, so
    `tau = 1` there by construction. It is reported as a reference line, not as
    evidence.

The load-bearing numbers are the two DART corpora, and the difficulty-level
table below, which needs no regression at all and cannot be broken by a
collinear covariate.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def poisson_fit(x: np.ndarray, c: np.ndarray, iters: int = 200):
    """Newton-Raphson for c ~ Poisson(exp(alpha + tau*x)), with a sandwich
    standard error because retained counts are heavily overdispersed."""
    design = np.column_stack([np.ones_like(x), x])
    beta = np.array([math.log(max(c.mean(), 1e-6)), 0.0])
    for _ in range(iters):
        mu = np.exp(np.clip(design @ beta, -30, 30))
        grad = design.T @ (c - mu)
        hess = -(design.T * mu) @ design
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        new = beta - step
        if not np.all(np.isfinite(new)):
            break
        if np.max(np.abs(new - beta)) < 1e-11:
            beta = new
            break
        beta = new
    mu = np.exp(np.clip(design @ beta, -30, 30))
    bread = np.linalg.inv((design.T * mu) @ design)
    meat = (design.T * (c - mu) ** 2) @ design
    sw = bread @ meat @ bread
    return beta[0], beta[1], math.sqrt(max(sw[1, 1], 0.0)), math.sqrt(max(bread[1, 1], 0.0))


def load_pool(pool_name: str):
    """Per query: retained (accepted) count, and the MATH difficulty level,
    which is an independent, human-annotated covariate."""
    from datasets import load_dataset

    ds = load_dataset(pool_name, split="train").select_columns(
        ["query", "ans_correct", "query_metadata"]
    )
    kept: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    level: dict[str, int] = {}
    for row in ds:
        q = row["query"]
        total[q] += 1
        kept[q] += bool(row["ans_correct"])
        meta = row.get("query_metadata")
        if isinstance(meta, dict) and "level" in meta and q not in level:
            try:
                level[q] = int(meta["level"])
            except (TypeError, ValueError):
                pass
    return kept, total, level


def retained_counts(name: str):
    from datasets import load_dataset

    ds = load_dataset(name, split="train").select_columns(["query"])
    counts: dict[str, int] = defaultdict(int)
    for row in ds:
        counts[row["query"]] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="hkust-nlp/dart-math-pool-math")
    ap.add_argument("--min-mass", type=int, default=2)
    ap.add_argument("--out", default="output/tau/dart.json")
    args = ap.parse_args()

    kept, total, level = load_pool(args.pool)
    n_all_accepted = sum(1 for q in total if kept[q] == total[q])
    print(f"[pool] {len(total)} queries, {sum(total.values())} rows")
    print(f"[pool] queries where every row is accepted: {n_all_accepted}/{len(total)}")
    print(f"[pool] queries with a difficulty level: {len(level)}")

    corpora = {"DART-Math-Uniform": "hkust-nlp/dart-math-uniform",
               "DART-Math-Hard (Prop2Diff)": "hkust-nlp/dart-math-hard"}
    counts = {"VRT (= the pool itself; definitional)": kept}
    for label, name in corpora.items():
        print(f"[corpus] {name}")
        counts[label] = retained_counts(name)

    queries = [q for q in kept if kept[q] >= args.min_mass]
    print(f"[join] {len(queries)} queries with pool mass >= {args.min_mass}")

    rows = []
    for label, cnt in counts.items():
        x = np.array([math.log(kept[q]) for q in queries])
        c = np.array([float(cnt.get(q, 0)) for q in queries])
        a, tau, rse, mse = poisson_fit(x, c)
        rows.append(
            {
                "corpus": label,
                "n_queries": len(queries),
                "mean_count": float(c.mean()),
                "tau": float(tau),
                "tau_robust_se": float(rse),
                "overdispersion": float(rse / mse) if mse > 0 else float("nan"),
                "definitional": label.startswith("VRT"),
            }
        )

    print(f"\n{'corpus':38s} {'mean c':>8s} {'tau':>7s} {'robust se':>10s} {'overdisp':>9s}")
    for r in rows:
        flag = "  (definitional)" if r["definitional"] else ""
        print(
            f"{r['corpus']:38s} {r['mean_count']:8.2f} {r['tau']:7.3f} "
            f"{r['tau_robust_se']:10.3f} {r['overdispersion']:9.1f}{flag}"
        )

    # --- the model-free view: mean retained count by difficulty level ------
    levels = sorted({level[q] for q in queries if q in level})
    if levels:
        print(f"\nmean retained count by MATH difficulty level (1 = easiest)")
        header = "corpus".ljust(38) + "".join(f"{f'L{v}':>9s}" for v in levels)
        print(header)
        by_level = {}
        for label, cnt in counts.items():
            cells = []
            for v in levels:
                qs = [q for q in queries if level.get(q) == v]
                cells.append(float(np.mean([cnt.get(q, 0) for q in qs])) if qs else float("nan"))
            by_level[label] = cells
            print(label.ljust(38) + "".join(f"{x:9.1f}" for x in cells))
        n_by_level = [sum(1 for q in queries if level.get(q) == v) for v in levels]
        print("queries".ljust(38) + "".join(f"{n:9d}" for n in n_by_level))
    else:
        by_level = {}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"pool": args.pool, "results": rows, "by_level": by_level,
                    "levels": levels}, indent=2)
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
