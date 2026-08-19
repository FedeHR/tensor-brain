# Note: final-layer normalization is a partial gauge fix

Recorded 2026-08-18 while setting up the `log Z` probe. This is my own
observation, not from the existing docs, and it sharpens brief claim 5.

## The observation

The Gaussian branch drops the term

    -(1/(2 sigma^2)) x^T W^T W x

and the exactness condition given in `tb_update_generalized.md` §8b is
`W^T W = c I`, a **tight frame**.

But that condition is stated over all of `R^n`. Real readouts do not see all of
`R^n`: every modern architecture applies **LayerNorm / RMSNorm immediately
before the readout**, so the state that reaches the index layer lies on (or very
near) a sphere `||x|| = R`.

Decompose

    W^T W = c I + S,      trace(S) = 0

with `c = trace(W^T W)/n`. On the sphere,

    x^T W^T W x = c R^2 + x^T S x

and `c R^2` is a **constant** — it contributes nothing to `Var[log Z]` and is
absorbed by the normalizing constant of the posterior.

> **Therefore the tight-frame condition is stronger than necessary. What the
> additive update actually requires is that the readout Gram matrix be
> isotropic *after projecting out its trace* — i.e. that `S = W^T W - cI` be
> small on the sphere the normalization layer puts the state on. Normalization
> makes the isotropic part free.**

## Why this matters

1. **It weakens the hypothesis in exactly the direction that makes it testable.**
   "Trained readouts form tight frames" is a strong claim that neural collapse
   supports only for balanced classification and that LLM unembeddings visibly
   violate (they are famously anisotropic). "Trained readouts are isotropic up
   to trace on the post-norm sphere" is much weaker and might actually hold.
   The measurable quantity becomes the **normalized frame defect**

       delta = ||W^T W - cI||_F / (c sqrt(n)),   c = trace(W^T W)/n

   which is scale-free and comparable across models and layers.

2. **It predicts why QK-norm and key L2-normalization help.** In the linear
   attention branch the same algebra applies to accumulated keys `K`. Per-key L2
   normalization fixes `||k_t||`, which fixes the *diagonal* of `K K^T` but not
   its off-diagonal structure. The relevant object is still the spread of the
   spectrum of `K K^T`. So the theory says key normalization is a partial fix
   and predicts a specific residual — the anisotropy of the key Gram — which is
   directly measurable in a trained checkpoint.

3. **It gives the categorical branch a matching statement.** For the softmax
   readout, `log Z(x) = logsumexp_k(a_{0,k} + a_k^T x)`. Its Hessian is
   `A (diag(pi) - pi pi^T) A^T`, the pi-weighted covariance of the embedding
   rows. The same trace-splitting applies on the post-norm sphere: only the
   *anisotropy* of the softmax-weighted embedding covariance is irreducible.
   This is a sharper, directly measurable version of the `K_eff` story: `K_eff`
   counts how many outcomes vote; this counts whether their votes are
   **isotropically distributed**.

## Consequence for the gauge

The affine gauge `A -> A - c 1^T` removes the *gradient* of `log Z`. The
observation above says normalization additionally removes the *trace of its
Hessian*. Together they say: the irreducible part of `log Z` is its
**traceless second-order-and-higher structure on the sphere**. That is a much
smaller object than `Var[log Z]` as measured naively, and it means naive
measurements of `Var[log Z]` on real models will **overstate** the error of the
additive update unless both are projected out.

This is directly testable and is the first thing the probe should report:
`Var[log Z]` decomposed into (constant) / (affine, = gauge) / (isotropic
quadratic, = free under normalization) / (irreducible remainder).
