"""Check: partial saliency gating gives a one-parameter correction family.

Window opens with prob pi(x) = Z(x)^tau / C, then k ~ softmax(a0 + A^T x).
Observing M fired windows, the exact posterior should be

    P(x | k_1..k_M)  proportional to  p(x) exp(sum_m a_km^T x) Z(x)^{M(tau-1)}

so the additive rule needs a correction of (1 - tau) * M * logZ(x).
tau = 1 -> plain additive update is exact.  tau = 0 -> the usual full correction.
"""

import itertools

import numpy as np

rng = np.random.default_rng(0)
DT = np.float64


def enumerate_states(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=DT)


def run(n=8, K=16, M=3, sigma=1.0, tau=0.5, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, sigma, size=(n, K))
    a0 = -np.logaddexp(0.0, A).sum(axis=0)  # column normalization
    q_prior = rng.normal(0.0, 0.7, size=n)

    X = enumerate_states(n)                     # [S, n]
    scores = a0[None, :] + X @ A                # [S, K]
    logZ = np.logaddexp.reduce(scores, axis=1)  # [S]
    log_cond = scores - logZ[:, None]           # log P(k|x)

    log_prior = (X * q_prior[None, :] - np.logaddexp(0.0, q_prior)[None, :]).sum(axis=1)

    ks = rng.integers(0, K, size=M)

    # Exact posterior under partial gating with exponent tau.
    # joint(x, k_1..k_M, all fired) ∝ p(x) * prod_m [ Z^tau/C * P(k_m|x) ]
    log_joint = log_prior + log_cond[:, ks].sum(axis=1) + M * tau * logZ
    log_joint -= log_joint.max()
    post = np.exp(log_joint)
    post /= post.sum()

    # Additive rule with a tau-scaled normalizer correction.
    q = q_prior + A[:, ks].sum(axis=1)
    log_add = (X * q[None, :]).sum(axis=1) - (1.0 - tau) * M * logZ
    log_add -= log_add.max()
    add = np.exp(log_add)
    add /= add.sum()

    kl = float((post * (np.log(post + 1e-300) - np.log(add + 1e-300))).sum())

    # For reference: the plain additive rule with no correction at all.
    log_plain = (X * q[None, :]).sum(axis=1)
    log_plain -= log_plain.max()
    plain = np.exp(log_plain)
    plain /= plain.sum()
    kl_plain = float((post * (np.log(post + 1e-300) - np.log(plain + 1e-300))).sum())
    return kl, kl_plain


print(f"{'tau':>5} {'KL(exact || tau-corrected)':>28} {'KL(exact || plain additive)':>29}")
for tau in [0.0, 0.25, 0.5, 0.75, 1.0]:
    kls, plains = [], []
    for seed in range(12):
        a, b = run(tau=tau, seed=seed)
        kls.append(a)
        plains.append(b)
    print(f"{tau:>5.2f} {np.mean(kls):>28.3e} {np.mean(plains):>29.4f}")
