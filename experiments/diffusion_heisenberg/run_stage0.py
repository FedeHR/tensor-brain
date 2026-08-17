"""Stage 0 of the diffusion-decoding assessment.

    PYTHONPATH=".:src" python -m experiments.diffusion_heisenberg.run_stage0 \
        --questions data/gsm8k/test.jsonl --num-questions 400

Measures what fraction of the true post-commit logit shift an additive,
state-independent correction captures. Three passes over the same seeded
trajectories:

    scan   which tokens greedy decoding commits, so a probe set can be chosen
    fit    accumulate the mean correction per probe token
    score  re-measure and evaluate leave-one-out, so nothing is scored on itself

If the additive rule captures little, the idea is dead and stages 1-3 are not
worth running -- which is the point of doing this first.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch

from . import probe as P


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--num-questions", type=int, default=400)
    parser.add_argument("--new-tokens", type=int, default=48)
    parser.add_argument("--steps", type=int, default=14, help="commits measured per question")
    parser.add_argument("--max-targets", type=int, default=10)
    parser.add_argument("--probe-tokens", type=int, default=60)
    parser.add_argument("--min-observations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("output/diffusion_heisenberg"))
    parser.add_argument("--model-cache", type=Path, default=Path("data/models"))
    parser.add_argument("--reuse-fit", action="store_true",
                        help="load the cached scan/fit instead of recomputing them")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    print(f"device {device}  dtype {dtype}", flush=True)

    model, tokenizer = P.load_model(args.model, device, dtype, cache=args.model_cache)
    mask_id = tokenizer.mask_token_id
    exclude = P.special_tokens(tokenizer)
    questions = P.load_questions(args.questions, args.num_questions)
    print(f"model {args.model}\n{len(questions)} questions, {args.new_tokens} masked "
          f"positions, {args.steps} commits measured each", flush=True)

    def walk(handler) -> None:
        for question in questions:
            prompt = P.build_prompt(tokenizer, question, device)
            x, block = P._prepare(prompt, mask_id, args.new_tokens)
            for event in P.trajectory(model, x, mask_id, block, steps=args.steps,
                                      max_targets=args.max_targets, exclude=exclude):
                handler(*event)

    # Scan and fit are deterministic given the questions and the model, so their
    # result is cached: scoring is the part worth iterating on.
    checkpoint = args.output / "fit.pt"
    counts_path = args.output / "counts.json"

    if args.reuse_fit and checkpoint.exists() and counts_path.exists():
        accumulators = P.Accumulators.load(checkpoint)
        counts = Counter({int(k): v for k, v in json.loads(counts_path.read_text()).items()})
        probe_tokens = accumulators.tokens
        print(f"reusing cached fit: {len(probe_tokens)} probe tokens, "
              f"{int(accumulators.count.sum())} corrections", flush=True)
    else:
        started = time.time()
        counts = Counter()
        walk(lambda token, *_: counts.update([token]))
        probe_tokens = [t for t, c in counts.most_common(args.probe_tokens)
                        if c >= args.min_observations]
        print(f"scan: {sum(counts.values())} commits, {len(counts)} distinct tokens, "
              f"{len(probe_tokens)} with >= {args.min_observations} observations "
              f"[{time.time() - started:.0f}s]", flush=True)
        if not probe_tokens:
            raise SystemExit("no token committed often enough; raise --num-questions")
        print("  most committed: " +
              ", ".join(f"{tokenizer.decode([t])!r}×{counts[t]}" for t in probe_tokens[:10]),
              flush=True)

        accumulators = P.Accumulators(vocab_size=model.config.vocab_size, tokens=probe_tokens)
        probe_only = set(probe_tokens)
        started = time.time()

        def accumulate(token, position, targets, before, after) -> None:
            if token not in probe_only:
                return
            for j in targets:
                accumulators.add(token, after[j] - before[j])

        walk(accumulate)
        print(f"fit: accumulated {int(accumulators.count.sum())} corrections "
              f"[{time.time() - started:.0f}s]", flush=True)
        args.output.mkdir(parents=True, exist_ok=True)
        accumulators.save(checkpoint)
        counts_path.write_text(json.dumps({str(k): v for k, v in counts.items()}))

    probe_set = set(probe_tokens)

    # ---- score ------------------------------------------------------------
    report = P.Report()
    embedding = P.embedding_matrix(model)
    free_cache: dict[int, torch.Tensor] = {}
    scales = torch.linspace(0.0, 1.5, 16)
    started = time.time()

    # per-scale KL sums, so a single global gain can be chosen honestly after the
    # fact rather than per sample -- the per-sample optimum is an oracle
    grid = {"additive": torch.zeros(len(scales)), "free": torch.zeros(len(scales))}
    grid_count = {"additive": 0, "free": 0}

    def score(token, position, targets, before, after) -> None:
        if token not in probe_set:
            return
        if token not in free_cache:
            free_cache[token] = P.free_correction(embedding, token)
        free = free_cache[token]
        for j in targets:
            reference = after[j].cpu()
            baseline = before[j].cpu()
            delta = reference - baseline

            do_nothing = float(P._kl(reference, baseline))
            if do_nothing < 1e-6:
                continue
            report.add("kl_do_nothing", do_nothing)
            report.add("offset", float(abs(j - position)))

            mean = accumulators.leave_one_out(token, delta)
            for name, direction in (("additive", mean), ("free", free)):
                if direction is None:
                    continue
                curve = P.scaled_kl_curve(reference, baseline, direction, scales)
                grid[name] += curve
                grid_count[name] += 1
                report.add(f"kl_{name}_oracle", float(curve.min()))
                report.add(f"{name}_oracle_scale", float(scales[int(curve.argmin())]))
                if name == "additive":
                    report.add("kl_additive_unscaled",
                               float(P._kl(reference, (baseline + mean).log_softmax(-1))))

    walk(score)
    print(f"score: [{time.time() - started:.0f}s]", flush=True)

    summary = report.summary()
    counts_by = {k: len(v) for k, v in report.rows.items()}
    base = summary["kl_do_nothing"]

    best_global = {}
    for name in grid:
        if grid_count[name]:
            curve = grid[name] / grid_count[name]
            index = int(curve.argmin())
            best_global[name] = {"scale": float(scales[index]), "kl": float(curve[index]),
                                 "curve": curve.tolist()}

    print(f"\nmeasured on {counts_by['kl_do_nothing']} (commit, target) pairs")
    print(f"{'rule':<40}{'mean KL, nats':>15}{'captured':>12}")
    print("-" * 67)
    print(f"{'do nothing (what decoders do now)':<40}{base:>15.4f}{'0.0%':>12}")
    rows = [("additive  q += a_k  (LOO, gain 1)", summary.get("kl_additive_unscaled"))]
    if "additive" in best_global:
        rows.append((f"additive  q += {best_global['additive']['scale']:.2f}·a_k  (global gain)",
                     best_global["additive"]["kl"]))
    if "free" in best_global:
        rows.append((f"free  {best_global['free']['scale']:.2f}·E E^T e_k  (global gain)",
                     best_global["free"]["kl"]))
    rows += [("additive, per-sample gain  (ORACLE)", summary.get("kl_additive_oracle")),
             ("free, per-sample gain  (ORACLE)", summary.get("kl_free_oracle"))]
    for label, value in rows:
        if value is not None:
            print(f"{label:<40}{value:>15.4f}{1 - value / base:>11.1%}")

    print(f"\nmean oracle gain: additive {summary.get('additive_oracle_scale', float('nan')):.3f}, "
          f"free {summary.get('free_oracle_scale', float('nan')):.3f}")
    print(f"mean target distance from the commit: {summary.get('offset', float('nan')):.1f} tokens")

    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    payload = {
        "config": config | {"device": device, "dtype": str(dtype)},
        "probe_tokens": [{"id": t, "text": tokenizer.decode([t]), "count": counts[t]}
                         for t in probe_tokens],
        "summary": summary,
        "counts": counts_by,
        "scales": scales.tolist(),
        "global_gain": best_global,
        "captured": {
            "additive_gain_1": 1 - summary["kl_additive_unscaled"] / base
            if "kl_additive_unscaled" in summary else None,
            "additive_global_gain": 1 - best_global["additive"]["kl"] / base
            if "additive" in best_global else None,
            "free_global_gain": 1 - best_global["free"]["kl"] / base
            if "free" in best_global else None,
            "additive_oracle_gain": 1 - summary["kl_additive_oracle"] / base
            if "kl_additive_oracle" in summary else None,
        },
    }
    destination = args.output / "stage0.json"
    destination.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
