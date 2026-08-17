"""Does the gate shape matter, or is it an artifact of the Z scale?

The QTB offset a0_k = -sum_i softplus(A_ik) normalizes each column over the
whole cube, which drives Z far below 1. A logistic gate Z/(1+Z) then behaves
almost like the full gate, because log(1+Z) ~ Z is nearly flat there. Rescale Z
by an offset shift and see whether that conclusion survives.
"""

import itertools

import numpy as np

DT = np.float64


def setup(n=8, K=16, sigma=1.0, seed=0, z_shift=0.0):
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, sigma, size=(n, K))
    a0 = -np.logaddexp(0.0, A).sum(axis=0) + z_shift
    q_prior = rng.normal(0.0, 0.7, size=n)
    X = np.array(list(itertools.product([0, 1], repeat=n)), dtype=DT)
    scores = a0[None, :] + X @ A
    logZ = np.logaddexp.reduce(scores, axis=1)
    log_cond = scores - logZ[:, None]
    log_prior = (X * q_prior[None, :] - np.logaddexp(0.0, q_prior)[None, :]).sum(axis=1)
    return A, X, logZ, log_cond, log_prior, q_prior, rng


def norm(lp):
    lp = lp - lp.max()
    p = np.exp(lp)
    return p / p.sum()


def kl(p, q):
    return float((p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum())


M = 3
print(f"{'E[Z]':>10} {'sd logZ':>9} {'sd log(1+Z)':>12} "
      f"{'plain|uncond':>13} {'plain|logistic':>15} {'plain|fullgate':>15}")
for z_shift in [0.0, 3.0, 6.0, 9.0]:
    ez, sdz, sd1z, u, lo, fu = [], [], [], [], [], []
    for seed in range(12):
        A, X, logZ, log_cond, log_prior, q_prior, rng = setup(seed=seed, z_shift=z_shift)
        w = norm(log_prior)
        ez.append(float((w * np.exp(logZ)).sum()))
        mz = (w * logZ).sum()
        sdz.append(float(np.sqrt((w * (logZ - mz) ** 2).sum())))
        l1z = np.logaddexp(0.0, logZ)
        m1 = (w * l1z).sum()
        sd1z.append(float(np.sqrt((w * (l1z - m1) ** 2).sum())))

        ks = rng.integers(0, A.shape[1], size=M)
        base = (X * (q_prior + A[:, ks].sum(axis=1))[None, :]).sum(axis=1)
        plain = norm(base)
        u.append(kl(norm(base - M * logZ), plain))
        lo.append(kl(norm(base - M * l1z), plain))
        fu.append(kl(norm(base), plain))
    print(f"{np.mean(ez):>10.3f} {np.mean(sdz):>9.3f} {np.mean(sd1z):>12.3f} "
          f"{np.mean(u):>13.4f} {np.mean(lo):>15.4f} {np.mean(fu):>15.4f}")
