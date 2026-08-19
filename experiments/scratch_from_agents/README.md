# Scratch scripts from the scoping agents

Three of the five research agents in the 2026-08-19 scoping pass stopped early
on an account spend limit. These are the probe scripts they had written but not
run. **None of them has been executed or verified.** They are kept because two
carry ideas worth chasing, both recorded in `docs/SYNTHESIS.md` §5:

- `logz_frontier.py` — the Stein-identity argument that the gauge direction is
  the *frequency-weighted* mean unembedding row (Zipfian Whitening, Yokoi et al.
  NeurIPS 2024) rather than the uniform mean ("all-but-the-top", Mu & Viswanath
  ICLR 2018). Also proposes checking the entropy/`log Z` correlation reported by
  Goldberger & Melamud (COLING 2018) for LSTMs.
- `tmp_cfg_probe.py` — the observation that token-level CFG renormalises at
  every step, so it equals the sequence-level geometric mixture tilted by
  `prod_t Z_t`, with sequence length in the role of `M`. This survives the
  correction that logit-space addition is exact within a step.
- `tmp_gauge_probe.py` — the seed of the gauge/OOD result, which I then built
  properly as `experiments/logz_geometry/gauge_ood.py`.
- `tmp_logz_probe.py` — an alternative `log Z` probe, superseded by
  `experiments/logz_geometry/probe.py`.

Treat anything in here as an unverified sketch, not as a result.
