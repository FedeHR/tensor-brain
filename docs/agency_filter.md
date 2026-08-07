# Memory Maze as a filter benchmark: what does the belief retain?

Fourth stage of the agency research line, and the first that trains no policy at all.

The three earlier stages ([`agency_results.md`](agency_results.md),
[`agency_minigrid.md`](agency_minigrid.md), [`agency_bayes.md`](agency_bayes.md)) all measured the
Tensor Brain through a reward signal, and all three ran into the same problem from different
directions: a reward score confounds the quantity under test with exploration, credit assignment
and seed luck. The Bayes-filter study made this explicit — the measurement update *has* the form of
a Bayes filter, but nothing in a policy gradient constrains the injected `a_k` to be a calibrated
log-likelihood ratio, and the test could not separate "the form is wrong" from "the training signal
never asked for it".

This study removes the policy. The claim under test is that

```
q <- alpha * q + beta * a_k
```

is a **recursive belief update**, so it is tested as one: replay a fixed recorded trajectory,
withhold observations at a controlled rate, and measure what the state still knows.

## 0. What a "filter" run actually does

No agent chooses anything. A scripted explorer is rendered once into a corpus and discarded. Every
model then replays that same recording, receiving at each step

- the action that was actually taken, and
- with probability `1 - rho`, the image that was actually seen.

It maintains a state, and is trained only to reconstruct the frame it is currently at. On an
observed step that is an autoencoder. On a **masked** step there is no image to copy, so the only
route to a reconstruction is the belief — which is where retention pressure comes from. At
`rho = 0.9` the model is blind for ten steps at a time on average.

The analogy is a blindfolded passenger who feels every turn and gets a glimpse every so often. We
are testing the passenger's mental map, not their driving.

Then the filter is frozen and probed against ground truth it never saw.

Two properties this buys that no earlier stage had:

1. **Every architecture sees byte-identical data.** A difference cannot be a difference in which
   states each policy happened to visit — the confound that damaged the gridworld comparison.
2. **Rendering is paid once.** MuJoCo runs at ~200 environment steps per second and does not
   parallelise inside a process, so an on-policy method re-renders on every run. A filter does not,
   which is what makes a 27-cell grid affordable at all.

## 1. Conditions

Nine, each answering one question. No cross-product: an ablation axis nobody will read is a run
that could have been a seed.

| condition | the question |
| --- | --- |
| `tb-none` | what does the index layer contribute at all? |
| `tb-raw` | the paper's update, `alpha = 1` |
| `tb-raw-alpha0` | does the earlier `alpha = 0` result **reverse** once retention is required? |
| `tb-corrected` | does a zero-mean write fix CBS saturation? |
| `tb-softplus-a0` | does QTB's log-normalizer bias (Eq. 31) do it instead? |
| `tb-soft` | does the *discreteness* of the measurement earn anything over attention? |
| `tb-raw-noact` | is the action write a control input? |
| `tb-accumulator` | is the bare log-odds filter — no learned prediction step — already enough? |
| `gru-control` | a capacity-comparable modern baseline |

Masking levels `rho` in `{0.0, 0.5, 0.9}`. **`rho = 0` is the control**: with full observation
nothing needs to be remembered, so the architectures should tie, and a gap there would indicate a
confound rather than memory.

`tb-raw-alpha0` is the condition most likely to be misread. Three earlier studies found `alpha = 0`
beating `alpha = 1`, and all three ran on tasks where nothing had to be retained between steps, so
discarding the prior cost nothing and avoided saturation. If that was a property of those tasks
rather than of the architecture, it should invert here. Either outcome is a result.

## 2. Capacity

The encoder is shared and identical, so every parameter of the difference sits after perception.

| condition | total | encoder | post-perception |
| --- | --- | --- | --- |
| `tb-*` | 309,984 | 268,128 | 41,856 |
| `tb-accumulator` | 277,088 | 268,128 | 8,960 |
| `gru-control` | 368,224 | 268,128 | 100,096 |

The control carries **more** capacity than the Tensor Brain, per the standing rule for this project.
`tb-accumulator` carries far less, and is a mechanism test rather than a fair baseline — it has no
learned prediction step by construction, and its number should be read that way.

## 3. Two corrections to the theory as originally proposed

**The drift correction cannot behave as predicted in the sharp regime.** The proposed
`q <- q + a_k - A p` was expected to help most when the index softmax is sharp. But as `p` sharpens
toward `delta_k`, the correction cancels the write entirely: `a_k - E_p[a] -> 0`, so `corrected`
degenerates to `none`. A test pins this. The defensible reading is different: `a_k - E_p[a]` is a
**zero-mean write**, a control variate that removes the systematic translation along `E_p[a]` which
drives `q` into saturation. That is a fix for the *diffuse* regime, and it targets a mechanism this
project has already measured (21% of units pinned at `alpha = 1`, 0% at `alpha = 0`). The logged
index entropy is what says which regime a run was actually in, and without it the ordering between
`none`, `raw` and `corrected` cannot be interpreted.

**There are two rival corrections, not one.** The control-variate write above and QTB's
log-normalizer bias `a_{0,k} = -sum_l softplus(a_{l,k})` are different answers to the same
normalisation problem, and the latter has already failed on two benchmarks under two optimizers.
Running both in one grid costs one config field and turns a standing "suspected discrepancy" into
an answer.

## 4. Measurements

1. **Linear probes** — ridge, closed form, from the state to `targets_pos`, `agent_pos`,
   `target_vec`. The benchmark's own protocol; the comparison a control can win. Closed form
   deliberately: a probe trained by gradient descent reports a property of the optimizer too.
2. **Memory-horizon curve** — the same probe conditioned on how many steps since that target was
   last in view. An averaged score cannot distinguish a filter that remembers from one reading the
   current frame; this can. It is the figure the study exists for.
3. **Native readout** — mutual information between the index the model spontaneously emits and
   which target is actually nearest. The bank is never supervised, so this asks whether the model's
   own vocabulary carved a real distinction. A GRU emits no symbol: not a worse number, **no
   measurement**. Reported as `null`, never as zero.
4. **Diagnostics** — index entropy, drift norm `||A p||`, saturated fraction, Monte-Carlo
   `Var[log Z]`, state norm. These are what turn an ordering into a mechanism.

Probes are fit on the `probe` split and scored on `test`, both held out from Phase 1 and built from
disjoint maze seeds. Adjacent steps within an episode are strongly correlated, so a random split
over pooled steps would leak and report a number that means nothing. Masking is applied at the rate
the filter was trained at — probing a `rho = 0.9` filter under full observation would measure a
different system from the one that was fit.

The colour probe's majority baseline sits near 0.89, because most steps have no target in view. It
is reported as a gap over that baseline; raw accuracy is not interpretable here.

## 5. The corpus

| | |
| --- | --- |
| splits | 576 train / 96 probe / 96 test episodes |
| episode | 1000 steps (the 9x9 level runs 250 s at 4 Hz control) |
| frame | 64x64x3 `uint8` = 12,288 bytes |
| size | ~9.4 GB |
| render time | ~10 minutes across 16 array tasks |

Behaviour is a **momentum random walk**, not a uniform random policy. The walker is a rolling ball
driven by torques, so redrawing every step averages the inputs to nearly zero and jitters in place;
a policy that never leaves its starting room puts every sample in the "never visible" bucket and the
memory-horizon curve has no domain. Actions are therefore held for a geometric dwell (mean 6 steps,
about 1.5 s).

Stored as `uint8` in per-field `.npy` files — `.npy` supports `mmap_mode` and `.npz` does not, so a
10 GB corpus is paged in on demand. Visibility is **not** stored: it depends on a field-of-view and
occlusion rule that are approximations, and baking an approximation into 10 GB would mean
re-rendering to revise it. The raw geometry is stored and visibility is derived in `horizon.py`.

## 6. Reproduce

```bash
# Cluster, in order. The second requires the first to have finished.
cluster/agency/setup.sh                      # on a COMPUTE node: uv sync + render smoke test
cluster/agency/submit_filter.sh corpus --partition=<name> --time=01:00:00
cluster/agency/submit_filter.sh grid   --partition=<name> --time=04:00:00
```

One cell by hand:

```bash
PYTHONPATH=src:. uv run --frozen --no-sync --python 3.12 \
  python -m experiments.agency.memorymaze.run_filter \
  --corpus "$MEMORYMAZE_CORPUS_ROOT" --condition tb-raw --mask 0.5 --seed 0
```

The grid stage renders nothing, so it needs no MuJoCo, no display and no EGL device — only a GPU.

## 7. Status

Implemented and tested: corpus recorder, scripted explorer, nine filter conditions, frame decoder,
Phase-1 training, all four Phase-2 measurements, visibility geometry, diagnostics, and both Slurm
stages. 40 tests in this file's module, 256 in the suite. The pipeline has been run end to end on a
tiny local corpus for all nine conditions.

**No result has been produced.** Nothing in this document is a finding, and the numbers in §2 are
parameter counts rather than outcomes.

## 8. Scope limits, stated in advance

- This measures **representation, not control**. A better filter is not automatically a better
  policy; representation-quality metrics frequently fail to predict control performance. The bridge
  is a separate run that warm-starts PPO from a fitted filter, and "does belief quality predict
  control?" is then its own question rather than an assumption.
- One seed per cell. This is a **screening pass** whose job is to select 3–4 conditions worth
  seeding properly, not a final ablation. Single seeds were meaningless in the RL stages because
  outcomes were bimodal on escape; there is no escape dynamic in a supervised fit on fixed data, and
  the `rho` sweep gives three points per condition, but the caveat stands.
- GRU is a comparable baseline, not a *modern* one for a belief-state paper. The right strengthening
  is an RSSM — the architecture Memory Maze was built to test. Deferred; the offline harness makes
  adding it cheap, since the data is fixed.
