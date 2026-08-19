"""The energy score is gauge-dependent.

Energy-based OOD detection (Liu et al., arXiv:2010.03759) scores an input by
``E(h) = -logsumexp(W h + b) = -log Z(h)`` and thresholds it: in-distribution
inputs are supposed to have higher total logit mass.

But the readout has an exact symmetry. For any vector ``c``,

    W <- W - 1 c^T          (subtract c from every row / class embedding)

shifts *every* logit by the same ``-c^T h``, so the softmax -- every predicted
probability, the argmax, the loss, the accuracy -- is **bit-identical**, while

    log Z(h)  ->  log Z(h) - c^T h,    i.e.   E(h) -> E(h) + c^T h.

Since ``c^T h`` varies across inputs, the energy score changes. So the detector's
score is not a function of the model; it is a function of the *coordinates the
model happens to be written in*.

This script makes that concrete on a real LM:

  * **baseline**   c = 0, the parameterisation training happened to land in
  * **flat gauge** c = the least-squares slope of log Z on h. This is the
    canonical "flattest" gauge, the one the Heisenberg update wants, and it
    should destroy the detector.
  * **adversarial** c chosen to *maximise* ID/OOD separation. A free detector
    improvement, obtained without touching the model.

If the AUROC moves while the softmax does not, the claim is established.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """P(score(pos) > score(neg)), computed by rank sum."""
    scores = torch.cat([pos, neg])
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.double)
    ranks[order] = torch.arange(len(scores), dtype=torch.double) + 1.0
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def states_for(model, tok, texts, n_states, seq_len):
    out_states = []
    total = 0
    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=seq_len)
        if ids["input_ids"].shape[1] < 8:
            continue
        h = model(**ids, output_hidden_states=True).hidden_states[-1][0]
        out_states.append(h.float())
        total += h.shape[0]
        if total >= n_states:
            break
    return torch.cat(out_states)[:n_states]


@torch.no_grad()
def random_token_states(model, tok, n_states, seq_len, seed=0):
    """OOD channel: uniformly random token ids. Same tokenizer, same length,
    no linguistic structure."""
    g = torch.Generator().manual_seed(seed)
    vocab = model.get_output_embeddings().weight.shape[0]
    out_states = []
    total = 0
    while total < n_states:
        ids = torch.randint(0, vocab, (1, seq_len), generator=g)
        h = model(input_ids=ids, output_hidden_states=True).hidden_states[-1][0]
        out_states.append(h.float())
        total += h.shape[0]
    return torch.cat(out_states)[:n_states]


def log_z(states, weight, bias, chunk=128):
    out = torch.empty(states.shape[0])
    for i in range(0, states.shape[0], chunk):
        logits = states[i : i + chunk] @ weight.T
        if bias is not None:
            logits = logits + bias
        out[i : i + chunk] = torch.logsumexp(logits, dim=-1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n-states", type=int, default=6000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--out", default="output/logz/gauge_ood_gpt2.json")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    weight = model.get_output_embeddings().weight.detach().float()
    head = model.get_output_embeddings()
    bias = head.bias.detach().float() if getattr(head, "bias", None) is not None else None

    ds = load_dataset("NeelNanda/pile-10k", split="train")
    texts = [t for t in ds["text"][:900] if len(t) > 500]

    id_states = states_for(model, tok, texts, args.n_states, args.seq_len)
    ood_states = random_token_states(model, tok, args.n_states, args.seq_len)
    print(f"[states] ID {tuple(id_states.shape)}  OOD {tuple(ood_states.shape)}")

    # --- the three gauges -------------------------------------------------
    # flat gauge: least-squares slope of log Z on h, fit on ID states only
    lz_id = log_z(id_states, weight, bias)
    design = torch.cat([torch.ones(len(id_states), 1), id_states], dim=1)
    c_flat = torch.linalg.lstsq(design, lz_id.unsqueeze(1)).solution.squeeze(1)[1:]

    # chosen gauge: a direction fit to separate the two clouds, on a TRAIN half,
    # scored on a HELD-OUT half. This does use OOD labels, so it is not a
    # deployable detector -- it is a demonstration that the score is not
    # identified by the model. Fisher/LDA direction on the shared covariance.
    n_id, n_ood = len(id_states), len(ood_states)
    tr_id, te_id = id_states[: n_id // 2], id_states[n_id // 2 :]
    tr_ood, te_ood = ood_states[: n_ood // 2], ood_states[n_ood // 2 :]
    mu_d = tr_ood.mean(0) - tr_id.mean(0)
    pooled = torch.cat([tr_id - tr_id.mean(0), tr_ood - tr_ood.mean(0)])
    cov = (pooled.T @ pooled) / len(pooled)
    w_lda = torch.linalg.solve(cov + 1e-3 * torch.eye(cov.shape[0]) * cov.diagonal().mean(), mu_d)
    w_lda = w_lda / w_lda.norm()

    gauges: dict[str, torch.Tensor] = {
        "baseline (c=0)": torch.zeros_like(c_flat),
        "flat gauge (LS slope, no OOD labels)": c_flat,
    }
    for alpha in (-30.0, -10.0, 10.0, 30.0):
        gauges[f"chosen gauge, alpha={alpha:+.0f}"] = alpha * w_lda

    results = []
    for name, c in gauges.items():
        w_g = weight - c.unsqueeze(0)
        # Conventional orientation: the OOD-ness score is the energy -log Z, so
        # a higher AUROC means a better detector.
        s_id = -log_z(te_id, w_g, bias)
        s_ood = -log_z(te_ood, w_g, bias)
        a = auroc(s_ood, s_id)

        # the model must be untouched
        probe = te_id[:64]
        p0 = torch.softmax(probe @ weight.T + (bias if bias is not None else 0.0), -1)
        p1 = torch.softmax(probe @ w_g.T + (bias if bias is not None else 0.0), -1)
        dp = (p0 - p1).abs().max().item()

        results.append(
            {
                "gauge": name,
                "auroc_energy_ood": a,
                "max_softmax_change": dp,
                "logz_var_id": float(s_id.var()),
            }
        )
        print(f"{name:38s} AUROC={a:6.4f}   max|Δsoftmax|={dp:.2e}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"model": args.model, "results": results}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
