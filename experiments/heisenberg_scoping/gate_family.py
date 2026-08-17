"""What correction does a general saliency gate require?

If a concept window opens with probability pi(x) = g(Z(x)) / C and, given that
it opened, k ~ softmax(a0 + A^T x), then over M fired windows

    P(x | k_1..k_M)  ∝  p(x) exp(sum_m a_km^T x) * [ g(Z(x)) / Z(x) ]^M

so the additive rule needs a correction of  M * [ log Z(x) - log g(Z(x)) ].

Check this for four gate shapes:
    g(Z) = Z^tau      -> correction (1-tau) M logZ      (tau=1 exact)
    g(Z) = Z/(1+Z)    -> correction M log(1+Z)          (logistic / outside option)
    g(Z) = 1          -> correction M logZ              (unconditional)
    g(Z) = 1{Z>c}     -> correction M logZ on the accepted region (hard threshold)
"""

import itertools

import numpy as np

DT = np.float64


def setup(n=8, K=16, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, sigma, size=(n, K))
    a0 = -np.logaddexp(0.0, A).sum(axis=0)
    q_prior = rng.normal(0.0, 0.7, size=n)
    X = np.array(list(itertools.product([0, 1], repeat=n)), dtype=DT)
    scores = a0[None, :] + X @ A
    logZ = np.logaddexp.reduce(scores, axis=1)
    log_cond = scores - logZ[:, None]
    log_prior = (X * q_prior[None, :] - np.logaddexp(0.0, q_prior)[None, :]).sum(axis=1)
    return A, X, logZ, log_cond, log_prior, q_prior, rng


def norm(logp):
    logp = logp - logp.max()
    p = np.exp(logp)
    return p / p.sum()


def kl(p, q):
    return float((p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum())


GATES = {
    "unconditional      g(Z)=1": (lambda lz: np.zeros_like(lz), lambda lz: lz),
    "full gate          g(Z)=Z": (lambda lz: lz, lambda lz: np.zeros_like(lz)),
    "partial gate    g(Z)=Z^0.5": (lambda lz: 0.5 * lz, lambda lz: 0.5 * lz),
    "logistic      g(Z)=Z/(1+Z)": (lambda lz: lz - np.logaddexp(0.0, lz), lambda lz: np.logaddexp(0.0, lz)),
}

M = 3
print(f"{'gate':<28} {'KL(exact||corrected)':>22} {'KL(exact||plain additive)':>27}")
for name, (log_g, correction) in GATES.items():
    corrected, plain = [], []
    for seed in range(12):
        A, X, logZ, log_cond, log_prior, q_prior, rng = setup(seed=seed)
        ks = rng.integers(0, A.shape[1], size=M)
        # exact posterior under this gate
        post = norm(log_prior + log_cond[:, ks].sum(axis=1) + M * log_g(logZ))
        q = q_prior + A[:, ks].sum(axis=1)
        base = (X * q[None, :]).sum(axis=1)
        corrected.append(kl(post, norm(base - M * correction(logZ))))
        plain.append(kl(post, norm(base)))
    print(f"{name:<28} {np.mean(corrected):>22.3e} {np.mean(plain):>27.4f}")
