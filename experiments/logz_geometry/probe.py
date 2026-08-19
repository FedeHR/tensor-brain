"""Measure the log-partition geometry of a real trained softmax readout.

The Heisenberg/additive update drops ``log Z(x) = logsumexp_k(a_{0,k} + a_k^T x)``
from the exact posterior. Its error is governed by ``Var[log Z]``, of which

  * the constant part is free (absorbed by normalisation),
  * the affine part is free -- an unidentified *gauge* of the readout:
    ``A -> A - c 1^T`` shifts every logit by the same ``c^T x``, so ``P(k|x)``
    is bit-identical while ``log Z`` loses its linear part,
  * only the remainder is irreducible, and it enters the error law as
    ``KL ~= (M^2/2) Var[r]``.

Nothing in the tensor-brain project has ever measured these on a real network.
This does, for any HF causal LM.

Two things make the measurement honest:

  * the gauge is **fit on one half of the states and scored on the other**.
    ``d`` is 768-2048, so an in-sample R^2 is an overfitting artifact.
  * the null controls **hold the state distribution fixed** and swap only the
    readout, for moment-matched random ones. Otherwise "log Z is affine" could
    be a property of the residual stream rather than of the trained readout.

Reading: the transformer residual stream is itself an additive accumulator --
every block writes into it and the unembedding reads it out, and nothing ever
subtracts log Z. So the affine fraction says how close that accumulation is to
being exactly Bayesian in the readout's own coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------


def load_texts(n_docs: int) -> list[str]:
    """Heterogeneous text. log Z geometry is only meaningful over a state
    distribution that actually spreads out, so a single-domain corpus (e.g.
    this repo's own markdown) would understate the variation badly."""
    from datasets import load_dataset

    ds = load_dataset("NeelNanda/pile-10k", split="train")
    return [t for t in ds["text"][: n_docs * 3] if len(t) > 500][:n_docs]


# --------------------------------------------------------------------------
# state collection
# --------------------------------------------------------------------------


@torch.no_grad()
def collect_states(model, tokenizer, texts, n_states, seq_len, device):
    """Hidden states entering the readout, i.e. *after* the final norm."""
    states = []
    total = 0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        ids = {k: v.to(device) for k, v in ids.items()}
        if ids["input_ids"].shape[1] < 8:
            continue
        out = model(**ids, output_hidden_states=True)
        states.append(out.hidden_states[-1][0].float().cpu())
        total += states[-1].shape[0]
        if total >= n_states:
            break
    if not states:
        raise RuntimeError("collected no hidden states")
    return torch.cat(states, dim=0)[:n_states]


def readout_matrix(model) -> tuple[torch.Tensor, torch.Tensor | None]:
    head = model.get_output_embeddings()
    weight = head.weight.detach().float().cpu()  # (V, d)
    bias = head.bias.detach().float().cpu() if getattr(head, "bias", None) is not None else None
    return weight, bias


# --------------------------------------------------------------------------
# null readouts -- state distribution held fixed, readout swapped
# --------------------------------------------------------------------------


def null_readouts(weight: torch.Tensor, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random readouts matched to the real one at increasing order.

    Note two transformations that are *not* useful controls because they are
    exact symmetries of log Z: permuting the rows of W (logsumexp is symmetric
    in k), and rotating rows and states together (W R, R^T x).
    """
    g = torch.Generator().manual_seed(seed)
    v, d = weight.shape
    mean = weight.mean(0)
    centred = weight - mean

    # second-moment matched: same row-cloud mean and covariance, Gaussian
    cov = (centred.T @ centred) / v
    evals, evecs = torch.linalg.eigh(cov.double())
    root = (evecs * evals.clamp_min(0).sqrt()) @ evecs.T
    gauss = torch.randn(v, d, generator=g).double() @ root.T + mean.double()

    # isotropic, matched only in total scale
    iso = torch.randn(v, d, generator=g) * centred.std() + mean

    # per-dimension marginals preserved, row structure destroyed
    shuffled = torch.stack([weight[torch.randperm(v, generator=g), j] for j in range(d)], dim=1)

    return {
        "gaussian_matched_cov": gauss.float(),
        "isotropic_matched_scale": iso,
        "column_shuffled": shuffled,
    }


def null_states(states: torch.Tensor, seed: int = 0) -> dict[str, torch.Tensor]:
    """The mirror-image control: hold the *readout* fixed and swap the state
    distribution. This separates "the trained readout has a flat log Z" from
    "the residual stream is confined to a manifold on which any log Z is flat".

    ``gaussian_matched_cov`` keeps the state covariance (hence its principal
    directions and effective dimension) but destroys everything higher-order;
    ``dim_shuffled`` keeps every coordinate's marginal but destroys the
    correlations, so its covariance is diagonal.
    """
    g = torch.Generator().manual_seed(seed + 1)
    n, d = states.shape
    mean = states.mean(0)
    centred = states - mean
    cov = (centred.T @ centred) / n
    evals, evecs = torch.linalg.eigh(cov.double())
    root = (evecs * evals.clamp_min(0).sqrt()) @ evecs.T
    gauss = (torch.randn(n, d, generator=g).double() @ root.T + mean.double()).float()
    shuffled = torch.stack([states[torch.randperm(n, generator=g), j] for j in range(d)], dim=1)
    return {"states_gaussian_matched_cov": gauss, "states_dim_shuffled": shuffled}


# --------------------------------------------------------------------------
# the decomposition
# --------------------------------------------------------------------------


@dataclass
class Geometry:
    model: str
    readout: str
    n_states: int
    dim: int
    vocab: int

    logz_var: float  # the raw object
    logz_sd: float
    affine_fraction: float  # held-out R^2 of log Z ~ 1 + x. Pure gauge.
    affine_fraction_null: float  # what a signal-free fit scores: about -d/n_train.
    # A held-out R^2 at or below this means log Z has NO affine
    # component; an in-sample fit would have reported +d/n instead.
    residual_var: float  # what the gauge fix cannot remove -> the error law
    residual_sd: float
    quadratic_fraction: float  # extra held-out R^2 from adding ||x||^2

    norm_cv: float  # is the state actually on a sphere?
    logit_sd: float
    k_eff_median: float  # 1 / sum_k pi_k^2, the effective vocabulary
    grad_norm_mean: float  # ||g(x)|| = ||A pi(x)||, the balance vector
    gauge_vs_mean_embed_cos: float  # is the gauge the "all-but-the-top" direction?
    gauge_norm: float


def logsumexp_stats(states, weight, bias, chunk=128):
    n, d = states.shape
    logz = torch.empty(n)
    k_eff = torch.empty(n)
    grads = torch.empty(n, d)
    s1 = s2 = 0.0
    count = 0
    for i in range(0, n, chunk):
        logits = states[i : i + chunk] @ weight.T
        if bias is not None:
            logits = logits + bias
        pi = torch.softmax(logits, dim=-1)
        logz[i : i + chunk] = torch.logsumexp(logits, dim=-1)
        k_eff[i : i + chunk] = 1.0 / (pi * pi).sum(-1)
        grads[i : i + chunk] = pi @ weight
        s1 += logits.sum().item()
        s2 += (logits * logits).sum().item()
        count += logits.numel()
    logit_sd = math.sqrt(max(s2 / count - (s1 / count) ** 2, 0.0))
    return logz, k_eff, grads, logit_sd


def analyse(model_name, readout_name, states, weight, bias, seed=0) -> Geometry:
    n, d = states.shape
    if n < 4 * d:
        raise ValueError(f"{n} states for d={d}: affine fit would be overfit; need >= {4 * d}")

    logz, k_eff, grads, logit_sd = logsumexp_stats(states, weight, bias)

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr, te = perm[: n // 2], perm[n // 2 :]

    def fit_score(cols):
        """Ridge path fit on train, scored on test. Plain least squares can go
        badly non-generalising when the state covariance is ill-conditioned,
        which would understate the affine part; the ridge path guards that
        without letting the fit see the test half."""
        xtr, xte = cols(states[tr]).double(), cols(states[te]).double()
        ytr = logz[tr].double()
        gram = xtr.T @ xtr
        rhs = xtr.T @ ytr
        scale = torch.diagonal(gram).mean()
        best = (None, float("inf"))
        for lam in (0.0, 1e-6, 1e-4, 1e-2, 1.0):
            reg = gram + lam * scale * torch.eye(gram.shape[0], dtype=gram.dtype)
            try:
                sol = torch.linalg.solve(reg, rhs)
            except RuntimeError:
                continue
            v = (logz[te].double() - xte @ sol).var(unbiased=False).item()
            if v < best[1]:
                best = (sol.float(), v)
        return best

    def affine(x):
        return torch.cat([torch.ones(x.shape[0], 1), x], dim=1)

    def affine_quad(x):
        return torch.cat([affine(x), (x * x).sum(-1, keepdim=True)], dim=1)

    var_te = logz[te].var(unbiased=False).item()
    sol, var_resid = fit_score(affine)
    c = sol[1:]  # the least-squares slope IS the gauge vector
    _, var_resid2 = fit_score(affine_quad)

    mean_embed = weight.mean(0)
    return Geometry(
        model=model_name,
        readout=readout_name,
        n_states=n,
        dim=d,
        vocab=weight.shape[0],
        logz_var=var_te,
        logz_sd=math.sqrt(var_te),
        affine_fraction=1.0 - var_resid / var_te if var_te > 0 else float("nan"),
        affine_fraction_null=-float(d) / float(len(tr)),
        residual_var=var_resid,
        residual_sd=math.sqrt(max(var_resid, 0.0)),
        quadratic_fraction=(var_resid - var_resid2) / var_te if var_te > 0 else float("nan"),
        norm_cv=(states.norm(dim=-1).std() / states.norm(dim=-1).mean()).item(),
        logit_sd=logit_sd,
        k_eff_median=k_eff.median().item(),
        grad_norm_mean=grads.norm(dim=-1).mean().item(),
        gauge_vs_mean_embed_cos=float((c @ mean_embed) / (c.norm() * mean_embed.norm() + 1e-12)),
        gauge_norm=c.norm().item(),
    )


def verify_gauge_is_free(states, weight, bias, c, n=64) -> tuple[float, float, float]:
    """The gauge fix must change nothing observable. Apply W <- W - 1 c^T and
    check the softmax is identical while log Z loses its linear part."""
    h = states[:n]
    logits = h @ weight.T + (bias if bias is not None else 0.0)
    logits_g = h @ (weight - c.unsqueeze(0)).T + (bias if bias is not None else 0.0)
    p, pg = torch.softmax(logits, -1), torch.softmax(logits_g, -1)
    lz, lzg = torch.logsumexp(logits, -1), torch.logsumexp(logits_g, -1)
    shift_err = (logits - logits_g - (h @ c).unsqueeze(1)).abs().max().item()
    return (p - pg).abs().max().item(), shift_err, (lzg.var() / lz.var()).item()


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n-states", type=int, default=20000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--n-docs", type=int, default=400)
    ap.add_argument("--nulls", action="store_true")
    ap.add_argument("--state-nulls", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()

    texts = load_texts(args.n_docs)
    print(f"[corpus] {len(texts)} pile documents")
    states = collect_states(model, tok, texts, args.n_states, args.seq_len, "cpu")
    weight, bias = readout_matrix(model)
    print(f"[states] {tuple(states.shape)}  readout {tuple(weight.shape)}")

    with torch.no_grad():
        ids = tok(texts[0], return_tensors="pt", truncation=True, max_length=64)
        out = model(**ids, output_hidden_states=True)
        rec = out.hidden_states[-1][0].float() @ weight.T
        if bias is not None:
            rec = rec + bias
        err = (rec - out.logits[0].float()).abs().max().item()
    print(f"[sanity] max |W h - logits| = {err:.3e}  (must be ~0, else the state is pre-norm)")
    if err > 1e-2:
        raise SystemExit("last hidden state is not the readout input; aborting")

    results = [analyse(args.model, "trained", states, weight, bias, args.seed)]
    if args.nulls:
        for name, w in null_readouts(weight, args.seed).items():
            results.append(analyse(args.model, name, states, w, None, args.seed))
    if args.state_nulls:
        for name, s in null_states(states, args.seed).items():
            results.append(analyse(args.model, name, s, weight, bias, args.seed))

    for g in results:
        print(f"\n=== {g.model} / readout={g.readout} ===")
        for k, v in asdict(g).items():
            if isinstance(v, float):
                print(f"  {k:28s} {v: .6g}")

    # the gauge must be free
    logz, _, _, _ = logsumexp_stats(states[:2048], weight, bias)
    des = torch.cat([torch.ones(2048, 1), states[:2048]], 1)
    c = torch.linalg.lstsq(des, logz.unsqueeze(1)).solution.squeeze(1)[1:]
    dp, shift, ratio = verify_gauge_is_free(states, weight, bias, c)
    print(f"\n[gauge] max |softmax change|      = {dp:.3e}   (must be ~0: the fix is free)")
    print(f"[gauge] max |logit shift - c^T x| = {shift:.3e}   (must be ~0)")
    print(f"[gauge] Var[logZ] after / before  = {ratio:.3e}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "results": [asdict(g) for g in results],
                    "gauge_check": {"softmax_delta": dp, "shift_err": shift, "var_ratio": ratio},
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
