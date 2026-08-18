"""Can tau be recovered from observable data?

Simulate a corpus where each instance has a latent x (OBSERVED at training time,
as ground-truth labels are), and the number of recorded observations is gated:

    count | x  ~  Poisson( lam * Z(x)^tau )

Then tau is the slope of log E[count | x] on log Z(x). Fit it by Poisson
regression and see whether the true tau comes back, and how many instances
are needed.
"""

import itertools

import numpy as np
from scipy.optimize import minimize

DT = np.float64


def build(n=10, K=40, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, sigma, size=(n, K))
    a0 = -np.logaddexp(0.0, A).sum(axis=0)
    q_prior = rng.normal(-0.5, 0.8, size=n)
    X = np.array(list(itertools.product([0, 1], repeat=n)), dtype=DT)
    logZ = np.logaddexp.reduce(a0[None, :] + X @ A, axis=1)
    return A, a0, q_prior, X, logZ


def corpus(q_prior, X, logZ, tau, num, mean_count, rng):
    """Draw instances from the prior and gated observation counts."""
    p = 1.0 / (1.0 + np.exp(-q_prior))
    bits = (rng.random((num, len(q_prior))) < p[None, :]).astype(np.int64)
    idx = bits @ (2 ** np.arange(len(q_prior) - 1, -1, -1))
    lz = logZ[idx]
    # normalize so the mean count is fixed regardless of tau
    rate = np.exp(tau * (lz - lz.mean()))
    rate = rate / rate.mean() * mean_count
    counts = rng.poisson(rate)
    return lz, counts


def fit_tau(lz, counts):
    """Poisson regression of counts on logZ; slope is tau."""
    z = lz - lz.mean()

    def nll(theta):
        b, t = theta
        eta = b + t * z
        return float(np.sum(np.exp(eta) - counts * eta))

    res = minimize(nll, x0=np.array([np.log(counts.mean() + 1e-9), 0.0]), method="BFGS")
    return float(res.x[1])


def main():
    A, a0, q_prior, X, logZ = build()
    print(f"spread of logZ over the prior: sd = {logZ.std():.3f}\n")
    print(f"{'true tau':>9} " + " ".join(f"{n:>12}" for n in [500, 2000, 10000, 50000]))
    for tau in [0.0, 0.25, 0.5, 0.75, 1.0]:
        row = []
        for num in [500, 2000, 10000, 50000]:
            ests = []
            for seed in range(20):
                rng = np.random.default_rng(1000 + seed)
                lz, counts = corpus(q_prior, X, logZ, tau, num, 4.0, rng)
                ests.append(fit_tau(lz, counts))
            row.append(f"{np.mean(ests):.3f}±{np.std(ests):.3f}")
        print(f"{tau:>9.2f} " + " ".join(f"{r:>12}" for r in row))


if __name__ == "__main__":
    main()
