"""Frontier additions to `experiments/logz_geometry/probe.py`.

`probe.py` holds the state distribution fixed and swaps the readout. The three
questions this file adds all go the other way, or go outside the readout:

  1. **Which gauge direction?**  The theory says the right `c` in `A -> A - c1^T`
     is the least-squares slope of `log Z` on `x`. Two named directions from the
     embedding literature compete with it: the *uniform* mean unembedding row
     (Mu & Viswanath's "all-but-the-top", ICLR 2018) and the *frequency-weighted*
     mean (Yokoi et al., "Zipfian Whitening", NeurIPS 2024). Stein's identity
     says `E[grad log Z] = E[A pi(x)] = A pbar`, i.e. the frequency-weighted one
     -- with the model's own predicted marginal as the frequency. Measured here.

  2. **Is flat `log Z` a property of the readout or of the state distribution?**
     Arora et al. (TACL 2016) Lemma 2.1 proves `Z_c` concentrates *for random
     unit-norm c* and verify it empirically (`Z_c` within [0.9, 1.1] of its
     mean). Real hidden states are not random directions. So: recompute
     `Var[log Z]` on random states matched to the real ones at increasing order,
     with the trained readout held fixed.

  3. **Does `log Z` track confidence?**  Goldberger & Melamud (COLING 2018)
     report Pearson r of -0.3..-0.64 between predictive entropy and `log Z_c`
     in LSTM LMs. If that reproduces in a transformer it makes the tilt
     `Z(x)^M` concrete: the additive update leans toward *confidently nameable*
     states, which is the energy score of Liu et al. (arXiv:2010.03759) read as
     a Bayesian bias rather than as an OOD statistic.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logz_geometry import probe  # noqa: E402

PILE = (
    "/Users/fede/.cache/huggingface/hub/datasets--NeelNanda--pile-10k/snapshots/"
    "127bfedcd5047750df5ccf3a12979a47bfa0bafa/data/train-00000-of-00001-4746b8785c874cc7.parquet"
)


def load_texts(n_docs: int) -> list[str]:
    import pyarrow.parquet as pq

    col = pq.read_table(PILE, columns=["text"]).column("text").to_pylist()
    return [t for t in col if len(t) > 500][:n_docs]


@torch.no_grad()
def collect(model, tok, texts, n_states, seq_len):
    """probe.collect_states, but also returning the realised token ids so the
    empirical unigram distribution can be built from the same sample."""
    states, ids_all, total = [], [], 0
    for text in texts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=seq_len)
        if enc["input_ids"].shape[1] < 8:
            continue
        out = model(**enc, output_hidden_states=True)
        states.append(out.hidden_states[-1][0].float())
        ids_all.append(enc["input_ids"][0])
        total += states[-1].shape[0]
        if total >= n_states:
            break
    return torch.cat(states)[:n_states], torch.cat(ids_all)[:n_states]


@torch.no_grad()
def marginal(states, weight, bias, chunk=128):
    """pbar = E_x[pi(x)], the model's own predicted marginal over the vocabulary,
    and the mean predictive entropy alongside."""
    acc = torch.zeros(weight.shape[0], dtype=torch.float64)
    ent = torch.empty(states.shape[0])
    for i in range(0, states.shape[0], chunk):
        logits = states[i : i + chunk] @ weight.T
        if bias is not None:
            logits = logits + bias
        logp = torch.log_softmax(logits, -1)
        pi = logp.exp()
        acc += pi.sum(0).double()
        ent[i : i + chunk] = -(pi * logp).sum(-1)
    return (acc / states.shape[0]).float(), ent


def gauge_scores(states, logz, weight, cand: dict[str, torch.Tensor], seed=0, rcond=1e-6):
    """Out-of-sample residual Var[log Z] after A <- A - c 1^T for each candidate
    c, plus the least-squares optimum. Only the *linear* part is being removed,
    so each candidate is scored with its own best intercept.

    Done in float64 with a SVD-based driver: the design matrix carries hidden
    states of norm O(100) next to a column of ones, and float32 `gels` gives a
    visibly different fit run to run on that conditioning.
    """
    n = states.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr, te = perm[: n // 2], perm[n // 2 :]
    x = states.double()
    y = logz.double()

    # LayerNorm before the readout puts the states on an exact affine subspace
    # (sum_i (x_i - b_i)/g_i = 0), so the design is rank-deficient by construction
    # and a plain lstsq inflates ||c|| along the null direction without changing
    # anything. Take the minimum-norm solution by truncated SVD.
    xm, ym = x[tr].mean(0), y[tr].mean()
    U, S, Vh = torch.linalg.svd(x[tr] - xm, full_matrices=False)
    keep = S > S[0] * rcond
    c_ls = Vh[keep].T @ ((U[:, keep].T @ (y[tr] - ym)) / S[keep])
    rank = int(keep.sum())

    cand = {"least_squares": c_ls, **{k: v.double() for k, v in cand.items()}}
    var0 = y[te].var(unbiased=False).item()
    out = {}
    for name, c in cand.items():
        resid = y - x @ c
        b = resid[tr].mean()  # its own best constant, fit on train
        rv = (resid[te] - b).var(unbiased=False).item()
        out[name] = {
            "residual_var": rv,
            "removed_fraction": 1 - rv / var0,
            "norm": c.norm().item(),
            "cos_to_least_squares": float(c @ c_ls / (c.norm() * c_ls.norm() + 1e-12)),
        }
    out['least_squares']['effective_rank'] = rank
    return var0, out, c_ls.float()


@torch.no_grad()
def state_nulls(states, seed=0):
    """Random states matched to the real ones at increasing order. The readout
    is held fixed; only the distribution the log-partition is evaluated over
    changes. `unit_sphere_matched_norm` is Arora et al.'s Lemma 2.1 setting."""
    g = torch.Generator().manual_seed(seed)
    n, d = states.shape
    mu = states.mean(0)
    ctr = states - mu
    cov = (ctr.T @ ctr) / n
    ev, U = torch.linalg.eigh(cov.double())
    root = (U * ev.clamp_min(0).sqrt()) @ U.T
    gauss = (torch.randn(n, d, generator=g).double() @ root.T).float() + mu

    r = states.norm(dim=-1, keepdim=True)
    sph = torch.randn(n, d, generator=g)
    sph = sph / sph.norm(dim=-1, keepdim=True) * r

    shuf = torch.stack([states[torch.randperm(n, generator=g), j] for j in range(d)], 1)
    return {
        "gaussian_matched_mean_cov": gauss,
        "sphere_matched_norm": sph,
        "coordinate_shuffled": shuf,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n-states", type=int, default=20000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--n-docs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    texts = load_texts(args.n_docs)
    states, ids = collect(model, tok, texts, args.n_states, args.seq_len)
    weight, bias = probe.readout_matrix(model)
    V, d = weight.shape
    print(f"[{args.model}] states {tuple(states.shape)}  readout {tuple(weight.shape)}")

    with torch.no_grad():
        enc = tok(texts[0], return_tensors="pt", truncation=True, max_length=64)
        out = model(**enc, output_hidden_states=True)
        rec = out.hidden_states[-1][0].float() @ weight.T
        if bias is not None:
            rec = rec + bias
        err = (rec - out.logits[0].float()).abs().max().item()
    print(f"[sanity] max |W h - logits| = {err:.3e}")
    if err > 1e-2:
        raise SystemExit("hidden_states[-1] is not the readout input (pre-norm); aborting")

    geom = probe.analyse(args.model, "trained", states, weight, bias, args.seed)
    logz, k_eff, grads, logit_sd = probe.logsumexp_stats(states, weight, bias)
    pbar, ent = marginal(states, weight, bias)

    emp = torch.zeros(V)
    for t, c in Counter(ids.tolist()).items():
        emp[t] = c
    emp /= emp.sum()

    cand = {
        "uniform_mean_embedding": weight.mean(0),  # Mu & Viswanath
        "model_marginal_weighted": weight.T @ pbar,  # E[grad log Z], Stein
        "corpus_freq_weighted": weight.T @ emp,  # Zipfian centering
        "mean_grad_log_Z": grads.mean(0),  # direct E[A pi(x)]
    }
    var0, gauges, c_ls = gauge_scores(states, logz, weight, cand, args.seed)

    nulls = {}
    for name, s in state_nulls(states, args.seed).items():
        lz, _, _, _ = probe.logsumexp_stats(s, weight, bias)
        des = torch.cat([torch.ones(s.shape[0], 1), s], 1)
        n = s.shape[0]
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
        tr, te = perm[: n // 2], perm[n // 2 :]
        des, lzd = des.double(), lz.double()
        sol = torch.linalg.lstsq(des[tr], lzd[tr].unsqueeze(1), driver="gelsd").solution.squeeze(1)
        rv = (lzd[te] - des[te] @ sol).var(unbiased=False).item()
        nulls[name] = {
            "logz_var": lz.var(unbiased=False).item(),
            "affine_fraction": 1 - rv / lz[te].var(unbiased=False).item(),
            "residual_var": rv,
        }

    ez = torch.corrcoef(torch.stack([ent, logz]))[0, 1].item()
    nz = torch.corrcoef(torch.stack([states.norm(dim=-1), logz]))[0, 1].item()

    res = {
        "model": args.model,
        "vocab": V,
        "dim": d,
        "n_states": states.shape[0],
        "logz_mean": logz.mean().item(),
        "logz_var": logz.var(unbiased=False).item(),
        "logz_sd": logz.std(unbiased=False).item(),
        "logz_range": [logz.min().item(), logz.max().item()],
        "affine_fraction_heldout": geom.affine_fraction,
        "quadratic_extra": geom.quadratic_fraction,
        "residual_var": geom.residual_var,
        "state_norm_cv": geom.norm_cv,
        "logit_sd": logit_sd,
        "k_eff_median": geom.k_eff_median,
        "grad_norm_mean": geom.grad_norm_mean,
        "corr_entropy_logz": ez,
        "corr_statenorm_logz": nz,
        "gauges": gauges,
        "state_nulls": nulls,
        "kl_error_law": {
            f"M={m}": {
                "raw": 0.5 * m * m * var0,
                "gauge_fixed": 0.5 * m * m * gauges["least_squares"]["residual_var"],
            }
            for m in (1, 2, 4)
        },
    }
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
