"""Which direction is the gauge?

The theory says the right `c` in `A -> A - c 1^T` is whatever flattens `log Z`.
Three named candidates compete, and they are NOT obviously the same vector:

  * the **least-squares slope** of `log Z` on the state -- what the theory
    literally asks for;
  * the **uniform mean unembedding row**, i.e. Mu & Viswanath's
    "all-but-the-top" (ICLR 2018);
  * the **frequency-weighted mean row** `A p_bar`. Stein's identity gives
    `E[grad log Z] = E[A pi(h)] = A p_bar`, so the *average gradient* of log Z
    is the mean row weighted by the model's own predicted marginal -- which is
    "Zipfian Whitening" (Yokoi et al., NeurIPS 2024).

The Stein argument makes the third look inevitable. This checks whether it is,
by measuring the residual variance each choice leaves behind. Written to verify
a claim from an unreviewed scratch script independently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe import collect_states, load_texts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n-states", type=int, default=20000)
    ap.add_argument("--out", default="output/logz/which_gauge_gpt2.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    head = model.get_output_embeddings()
    weight = head.weight.detach().float()
    bias = head.bias.detach().float() if getattr(head, "bias", None) is not None else None

    states = collect_states(model, tok, load_texts(300), args.n_states, 512, "cpu")
    n = len(states)

    # log Z and the mean gradient E[A pi(h)], chunked
    logz = torch.empty(n)
    grad_sum = torch.zeros(states.shape[1])
    for i in range(0, n, 128):
        logits = states[i : i + 128] @ weight.T
        if bias is not None:
            logits = logits + bias
        pi = torch.softmax(logits, -1)
        logz[i : i + 128] = torch.logsumexp(logits, -1)
        grad_sum += (pi @ weight).sum(0)
    mean_grad = grad_sum / n

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    tr, te = perm[: n // 2], perm[n // 2 :]

    # least-squares slope, fit on train
    design = torch.cat([torch.ones(len(tr), 1), states[tr]], 1).double()
    sol = torch.linalg.lstsq(design, logz[tr].double().unsqueeze(1)).solution.squeeze(1)
    c_ls = sol[1:].float()

    candidates = {
        "least_squares_slope": c_ls,
        "uniform_mean_row (all-but-the-top)": weight.mean(0),
        "marginal_weighted_row (Stein / Zipfian)": mean_grad,
        "none (c=0)": torch.zeros_like(c_ls),
    }

    var_te = logz[te].var(unbiased=False).item()
    rows = []
    for name, c in candidates.items():
        # after the gauge, log Z'(h) = log Z(h) - c.h ; the best remaining
        # constant is free, so score the variance of the residual.
        resid = logz[te] - states[te] @ c
        v = resid.var(unbiased=False).item()
        cos = float((c @ c_ls) / (c.norm() * c_ls.norm() + 1e-12)) if c.norm() > 0 else 0.0
        rows.append(
            {
                "gauge": name,
                "residual_var": v,
                "removed_fraction": 1.0 - v / var_te,
                "norm": float(c.norm()),
                "cos_to_least_squares": cos,
            }
        )
        print(
            f"{name:42s} residual_var={v:10.3f}  removed={1 - v / var_te:8.5f}  "
            f"|c|={float(c.norm()):8.3f}  cos_to_LS={cos:+.4f}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"model": args.model, "logz_var_heldout": var_te, "gauges": rows}, indent=2)
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
