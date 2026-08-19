"""Measure the per-step CFG log-normalizer  log Z_t = log sum_k p_u^(1-w) p_c^w.

Token-level CFG samples  P~(y) = prod_t [p_u^(1-w) p_c^w / Z_t]
                              = [p_u(y)^(1-w) p_c(y)^w] / prod_t Z_t
whereas the sequence-level geometric mixture is the same numerator / one global Z.
So token-level CFG = sequence-level target tilted by prod_t Z_t^{-1}.
This script measures how large that tilt is, and how much it varies across
sequences, on a laptop-scale model.
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
W = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
NTOK = 60
NSEQ = 8

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
m.eval()
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

PROMPTS = [
    "Write a short poem about the sea.\n",
    "Explain why the sky is blue.\n",
    "List three uses of a hammer.\n",
    "Translate to French: the cat sits on the mat.\n",
]
NEG = ""  # unconditional branch = empty context (the Sanchez et al. convention)

torch.manual_seed(0)
rows = []
for p in PROMPTS:
    for s in range(NSEQ):
        cid = tok(p, return_tensors="pt").input_ids
        uid = tok(NEG if NEG else tok.bos_token, return_tensors="pt").input_ids
        logZs, dis = [], []
        for t in range(NTOK):
            with torch.no_grad():
                lc = m(cid).logits[0, -1]
                lu = m(uid).logits[0, -1]
            pc = F.log_softmax(lc, -1)
            pu = F.log_softmax(lu, -1)
            mix = (1 - W) * pu + W * pc  # log of unnormalized p_u^(1-w) p_c^w
            logZ = torch.logsumexp(mix, -1).item()  # the per-step normalizer, in nats
            logZs.append(logZ)
            # Renyi identity check: log Z = (w-1) * D_w(p_c || p_u)
            dis.append(logZ / (W - 1) if W != 1 else 0.0)
            nxt = torch.multinomial(torch.softmax(mix, -1), 1)
            cid = torch.cat([cid, nxt[None]], 1)
            uid = torch.cat([uid, nxt[None]], 1)
        lz = np.array(logZs)
        rows.append((p[:28], lz.sum(), lz.mean(), lz.var(), np.mean(dis)))

print(f"model={MODEL}  w={W}  {NTOK} tokens x {len(rows)} sequences")
print(f"{'prompt':30s} {'sum logZ':>10s} {'mean':>8s} {'var/step':>9s} {'D_w':>7s}")
for r in rows:
    print(f"{r[0]:30s} {r[1]:10.2f} {r[2]:8.3f} {r[3]:9.4f} {r[4]:7.3f}")
S = np.array([r[1] for r in rows])
print("\nsum_t log Z_t over a %d-token sequence: mean %.2f nats, sd ACROSS SEQUENCES %.2f nats" % (NTOK, S.mean(), S.std()))
print("=> token-level CFG re-weights sequences by exp(-sum_t log Z_t); a %.1f-nat spread" % S.std())
print("   is a factor e^%.1f = %.3g in relative sequence probability vs the sequence-level target." % (S.std(), np.exp(S.std())))
