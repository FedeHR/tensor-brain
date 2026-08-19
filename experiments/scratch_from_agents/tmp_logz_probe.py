import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
m.eval()

# a heterogeneous corpus: prose, code, sql, math, noise, non-English, clinical
CHUNKS = [
    "The capital of France is Paris, a city known for its museums and its river. "
    "Tourists arrive every year to walk along the Seine and to visit the Louvre, "
    "which houses the Mona Lisa and thousands of other works of art. ",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n"
    "class Node:\n    def __init__(self, value, left=None, right=None):\n        self.value = value\n"
    "        self.left = left\n        self.right = right\n\n    def insert(self, v):\n"
    "        if v < self.value:\n            if self.left is None:\n                self.left = Node(v)\n"
    "            else:\n                self.left.insert(v)\n",
    "Once upon a time, in a small village at the edge of a dark forest, there lived a girl "
    "who could speak to birds. Every morning she went to the well, and every evening the birds "
    "told her what they had seen from the sky. One winter the well froze, and the birds "
    "stopped coming, and the girl set out to find them. ",
    "SELECT customer_id, SUM(total) FROM orders WHERE created_at > '2024-01-01' GROUP BY customer_id "
    "HAVING SUM(total) > 1000 ORDER BY 2 DESC LIMIT 50;\n"
    "CREATE INDEX idx_orders_created ON orders (created_at, customer_id);\n"
    "WITH monthly AS (SELECT date_trunc('month', created_at) m, count(*) c FROM orders GROUP BY 1) "
    "SELECT * FROM monthly WHERE c > 100;\n",
    "In quantum mechanics, the Heisenberg uncertainty principle states that the position and the "
    "momentum of a particle cannot both be known to arbitrary precision. Formally, the product of "
    "the standard deviations is bounded below by hbar over two. This is a consequence of the "
    "non-commutativity of the corresponding operators, not of any experimental limitation. ",
    "xkq zzr ,, ;; 91827 !!! qqq vvv --- 000 ??? ~~~ ### @@@ %%% ^^^ &&& *** ((( ))) ___ +++ === "
    "aaaa bbbb cccc dddd 5 5 5 5 5 zzzzzzzz ,,,,,,, ....... ;;;;;;; ",
    "The patient presented with acute chest pain radiating to the left arm and diaphoresis. "
    "An ECG showed ST elevation in leads II, III and aVF. Troponin was elevated at 4.2 ng/mL. "
    "The patient was taken for emergent catheterization and a stent was placed in the right "
    "coronary artery. Post-procedure the patient was started on dual antiplatelet therapy. ",
    "Ich habe gestern einen langen Spaziergang durch den Park gemacht und dabei viel nachgedacht. "
    "Der Herbst ist meine liebste Jahreszeit, weil die Blaetter sich verfaerben und die Luft klar wird. "
    "Am Abend habe ich mich an den Schreibtisch gesetzt und einen Brief geschrieben. ",
    "Theorem 1. Let f be continuous on [a,b] and differentiable on (a,b). Then there exists c in (a,b) "
    "such that f(b) - f(a) = f'(c)(b - a). Proof. Define g(x) = f(x) - f(a) - (x-a)(f(b)-f(a))/(b-a). "
    "Then g(a) = g(b) = 0, and by Rolle's theorem there is c with g'(c) = 0, which gives the result. ",
    "Q: What is the boiling point of water at sea level?\nA: 100 degrees Celsius.\n"
    "Q: Who wrote Pride and Prejudice?\nA: Jane Austen.\n"
    "Q: What is the largest planet in the solar system?\nA: Jupiter.\n"
    "Q: In what year did the Berlin Wall fall?\nA: 1989.\n",
]
TEXTS = [c * 4 for c in CHUNKS]  # lengthen so we clear the d-dimensional fit

H, LZ, GRP = [], [], []
for gi, t in enumerate(TEXTS):
    ids = tok(t, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        out = m(**ids, output_hidden_states=True)
    h = out.hidden_states[-1][0]
    lz = torch.logsumexp(out.logits[0], dim=-1)
    H.append(h)
    LZ.append(lz)
    GRP.append(np.full(len(lz), gi))

H = torch.cat(H).float().numpy()
LZ = torch.cat(LZ).float().numpy()
GRP = np.concatenate(GRP)
n, d = H.shape
print(f"model={MODEL}  tokens={n}  d={d}  V={m.config.vocab_size}")
print("log Z:  mean %.3f  sd %.3f  Var %.3f  range [%.2f, %.2f]" % (LZ.mean(), LZ.std(), LZ.var(), LZ.min(), LZ.max()))

# between-domain vs within-domain variance of log Z
gm = np.array([LZ[GRP == g].mean() for g in range(len(TEXTS))])
print("between-domain Var[E logZ] = %.3f  (%.0f%% of total)" % (gm.var(), 100 * gm.var() / LZ.var()))

# held-out affine fit: is log Z affine in the hidden state? (the GAUGE part)
rng = np.random.default_rng(0)
perm = rng.permutation(n)
ntr = int(0.7 * n)
tr, te = perm[:ntr], perm[ntr:]
X = np.concatenate([H, np.ones((n, 1))], 1)
coef, *_ = np.linalg.lstsq(X[tr], LZ[tr], rcond=None)
res_te = LZ[te] - X[te] @ coef
print("held-out affine fit (ntr=%d, d=%d): R^2 = %.4f   Var[resid] = %.4f" % (ntr, d, 1 - res_te.var() / LZ[te].var(), res_te.var()))

# ridge, to be fair when n is not >> d
for lam in [1e-2, 1e0, 1e2, 1e4]:
    A = X[tr].T @ X[tr] + lam * np.eye(d + 1)
    cf = np.linalg.solve(A, X[tr].T @ LZ[tr])
    r = LZ[te] - X[te] @ cf
    print("   ridge lam=%-7g  R^2_test = %.4f  Var[resid] = %.4f" % (lam, 1 - r.var() / LZ[te].var(), r.var()))

c = coef[:-1]
W = m.get_output_embeddings().weight.detach().float().numpy()
print("||c|| = %.3f ;  mean ||a_k|| = %.3f ;  ||mean unembed row|| = %.3f" % (np.linalg.norm(c), np.linalg.norm(W, axis=1).mean(), np.linalg.norm(W.mean(0))))
print("cos(c, mean unembed row) = %.3f" % (c @ W.mean(0) / (np.linalg.norm(c) * np.linalg.norm(W.mean(0)) + 1e-12)))
