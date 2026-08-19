"""Gauge experiment on a real LM.

Gauge:  W_U <- W_U - 1 c^T ,  b <- b - d      (every logit shifts by -(c.h + d))
        => softmax(logits) is BIT-IDENTICAL; log Z(h) -> log Z(h) - c.h - d.

Q1: how much of Var_h[log Z] is affine in h (held out)?
Q2: is the least-squares slope c the mean unembedding row (the classic
    `center_unembed` choice) or the SOFTMAX-WEIGHTED mean row  E_h[A pi(h)]?
Q3: does the gauge fix destroy an energy-score (-logsumexp) OOD detector
    while leaving every predicted probability unchanged?
"""

import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
m.eval()

ID_TEXT = (
    "The capital of France is Paris, a city known for its museums and its river. Tourists arrive "
    "every year to walk along the Seine and to visit the Louvre. In the summer the gardens are full "
    "of people reading on the grass, and in the winter the streets are quiet and grey. The history "
    "of the city runs back two thousand years, and much of it is still visible in the stones. "
    "Many writers have lived here, and many have left. The river divides the city into two halves "
    "that have never quite agreed with one another about anything at all. "
) * 8

OOD_TEXT = (
    "xkq zzr ,, ;; 91827 !!! qqq vvv --- 000 ??? ~~~ ### @@@ %%% ^^^ &&& *** ((( ))) ___ +++ === "
    "zzzz 5 5 5 5 wq wq wq ;;;;;; ...... ,,,,,, jjj kkk lll 88888 !!!!!! ????? ~~~~~ ##### "
) * 12


def collect(text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        out = m(**ids, output_hidden_states=True)
    return out.hidden_states[-1][0].float(), out.logits[0].float()


Hid, Lid = collect(ID_TEXT)
Hood, Lood = collect(OOD_TEXT)
d = Hid.shape[1]
lz_id = torch.logsumexp(Lid, -1).numpy()
lz_ood = torch.logsumexp(Lood, -1).numpy()
Hid_n, Hood_n = Hid.numpy(), Hood.numpy()

# ---- fit the gauge on a TRAIN half of the ID stream only
n = len(Hid_n)
tr, te = np.arange(n // 2), np.arange(n // 2, n)
X = np.concatenate([Hid_n, np.ones((n, 1))], 1)
lam = 1.0
A = X[tr].T @ X[tr] + lam * np.eye(d + 1)
coef = np.linalg.solve(A, X[tr].T @ lz_id[tr])
c, d0 = coef[:-1], coef[-1]

res_te = lz_id[te] - X[te] @ coef
print(f"model={MODEL}  d={d}  n_id={n}  n_ood={len(Hood_n)}")
print("Var[logZ] on held-out ID = %.3f ; after affine (gauge) fit = %.4f  -> %.2f%% removed"
      % (lz_id[te].var(), res_te.var(), 100 * (1 - res_te.var() / lz_id[te].var())))

# ---- apply the gauge to the actual weights and check the softmax is unchanged
W = m.get_output_embeddings().weight.detach().clone()
Wg = W - torch.tensor(c, dtype=W.dtype)[None, :]
p_before = torch.softmax(Hid @ W.T, -1)
p_after = torch.softmax(Hid @ Wg.T, -1)
print("max |p_before - p_after| over all tokens/positions = %.3e   (gauge is unobservable)"
      % (p_before - p_after).abs().max().item())

# ---- gauge direction: mean row vs softmax-weighted mean row
mean_row = W.mean(0).numpy()
pi_bar = p_before.mean(0).numpy()          # average next-token distribution on ID text
sw_row = (pi_bar[:, None] * W.numpy()).sum(0)   # E[A pi(h)] -- the true grad of logZ
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
print("cos(c, mean unembed row)            = %+.3f   <- the `center_unembed` choice" % cos(c, mean_row))
print("cos(c, softmax-weighted mean row)   = %+.3f   <- E_h[A pi(h)], the true dlogZ/dh" % cos(c, sw_row))
print("||c|| = %.3f  ||mean row|| = %.3f  ||sw row|| = %.3f"
      % (np.linalg.norm(c), np.linalg.norm(mean_row), np.linalg.norm(sw_row)))

# ---- does the gauge fix kill the energy-score OOD detector?
def auroc(pos, neg):
    lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos, nneg = len(pos), len(neg)
    return (r[lab == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

e_id_raw, e_ood_raw = lz_id[te], lz_ood
e_id_g = lz_id[te] - (Hid_n[te] @ c + d0)
e_ood_g = lz_ood - (Hood_n @ c + d0)
print("\nenergy score (+logsumexp), ID vs OOD:")
print("  raw gauge   : AUROC = %.3f   (mean ID %.2f vs OOD %.2f)" % (auroc(e_id_raw, e_ood_raw), e_id_raw.mean(), e_ood_raw.mean()))
print("  flat gauge  : AUROC = %.3f   (mean ID %.2f vs OOD %.2f)" % (auroc(e_id_g, e_ood_g), e_id_g.mean(), e_ood_g.mean()))
print("  (the model's predictions are identical in both rows)")

# entropy, a gauge-invariant baseline
H_id = -(p_before[te] * torch.log(p_before[te] + 1e-12)).sum(-1).numpy()
p_ood = torch.softmax(Hood @ W.T, -1)
H_ood = -(p_ood * torch.log(p_ood + 1e-12)).sum(-1).numpy()
print("  entropy (gauge-invariant): AUROC = %.3f" % auroc(-H_id, -H_ood))
