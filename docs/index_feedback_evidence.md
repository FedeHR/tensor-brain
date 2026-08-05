# What the papers claim about the index layer, and what our pair experiments actually tested

Written after the corrected pair runs (`runs/pair-corrected-original-seed0`,
`runs/pair-corrected-qtb-seed0`), both of which found identity feedback null.
Purpose: separate the architectural claim (the index layer is the core of the TB)
from the empirical claim our experiments were built to test, and identify which
experiment would actually settle the latter.

## 1. The index layer is the core, and our experiments confirm it is load-bearing

This is not in dispute and is not what the null result is about.

The original paper's architecture is the bilayer tensor network: a symbolic
**index layer** and a subsymbolic **representation layer**, with the dynamic
context layer as a third component (tb_original §1, p.3). The QTB draft states
the principle directly (qtb_LATEST p.10):

> - Information processing in the brain emerges from the interaction of two
>   layers: the subsymbolic representation layer and the symbolic index layer.
> - **Top-down inference is essential for both memory functions and symbol
>   grounding.**

`docs/fidelity.md:57` records the corresponding implementation commitment: the
same `A` serves bottom-up scoring and top-down feedback, and for feedback it is
imperative that `A` is used directly.

**Our corrected models are entirely index-layer models.** Every prediction —
predicate, both identities, and all ten category readouts — is a bottom-up index
score `sigmoid(q)ᵀA[:, k]` through the single shared global `A` with 2,783
columns. The corrected QTB run reaches 55.5% R@1, KL 1.674 and the best
unseen-triple R@1 of any model in the project *through that mechanism*. §5 of the
QTB analysis is also a result about the index layer: the QTB log-normalizer
`a₀ₖ = −Σ_l softplus(a_{l,k})` is what makes the readout data-driven rather than
prior-driven, and it is worth ~5 pp R@1.

So the index layer is tested continuously and it works. What came out null is one
specific pathway: **top-down injection of *identity* index embeddings into `q`
during the pair schedule, evaluated on novel entities.**

## 2. The paper's binary-label evidence does not isolate index feedback

This is the substantive finding, and it revises how the previous rounds framed
the experiment.

The paper's binary-label result (Table 4, VRD-E):

| Model | @10 | @1 |
|---|---:|---:|
| P-Direct | 85.45 | 31.68 |
| P-Samp | 90.39 | 45.09 |
| P-SA | 91.33 | **46.84** |

A +15.16 pp @1 gap, which earlier rounds treated as the paper's evidence that
index feedback is crucial for binary labels. But **P-Direct removes two things at
once.** From §6.3 and the Table 5 caption:

> In direct perception ... There are connections from the representation layer to
> the index layer but not in the opposite direction. **Also, there are no
> connections between the representation layer and the dynamic context layer.**

> In P-Direct, there are no links from `n` to `q`, **and `q` and `h` are
> independent.**

And the paper attributes the binary-label gap to the *dynamic context layer*,
three separate times, never to index feedback:

> The inferior performance of P-Direct confirms that **the dynamic context layer
> is important** for achieving good performance. (§6.5)

> As the inferior results for P-direct indicate, **the dynamic context layer is
> essential** to perform well in binary label prediction. (Table 4 caption)

> The much better performance of P-SA compared to P-Direct **demonstrates the
> importance of the dynamic context layer.** (Table 6 caption)

The paper therefore never ran the one-factor feedback ablation for binary labels.
**Our corrected experiment did** — Integral no-feedback vs Integral P-SA, with the
identical evolution schedule in both arms — and it is a cleaner ablation than the
paper's. Finding it null does not contradict any claim the paper makes.

## 3. Where the paper does isolate feedback on novel entities, it is also near-null

Unary labels, VRD-E (novel entities in test), Table 5:

| Model | Average over label groups |
|---|---:|
| P-Direct | 77.97 |
| P-Samp | 77.26 |
| P-SA | **78.09** |

P-SA beats P-Direct by **+0.12 points**, and P-Samp is *worse* than P-Direct. The
paper's own comment: "Not surprisingly, P-Direct is also quite competitive."

Our corrected pair result (P-SA − no-feedback = +8 assignments in one array, +3 in
the other, sign-inconsistent) is qualitatively the same outcome as the paper's own
novel-entity numbers.

## 4. Where feedback *is* decisive in the paper: known entities

Unary labels, VRD-EX, Table 5:

| Model | Entity | Average |
|---|---:|---:|
| P-Direct | 92.20 | 89.56 |
| P-Samp | 92.81 | **95.93** |
| P-SA | 92.65 | 95.12 |

P-Samp gains **+6.37 points** over P-Direct. The paper's explanation:

> On the VRD-EX data set with known entities in testing, perception with sampling,
> P-Samp, is best, where the algorithm could "remember" past encounters of the same
> entities. Here, P-Direct is not competitive.

> P-Direct is significantly worse since it cannot benefit from memory.

> For known entities (VRD-EX), entity indices permit some memorization and improve
> performance. (Table 4 caption)

VRD-EX is constructed by *distorting the training images* (p.23), so the same
entity instances appear in training and test. That is the regime in which the
identity bank carries usable information at evaluation time.

## 5. Our pair setup is the VRD-E regime, and more extreme

`docs/fidelity.md:97` — each protocol supplies exactly the identities supervised by
its own training records. `pair_experiment.py:623-626` loads
`heldout_video/train_pairs.jsonl` for training and
`heldout_video/development_pairs.jsonl` for evaluation, and the split reserves
whole *videos* (`fidelity.md:128`).

So at development time:

- the identity group holds 2,474 **training** identities;
- development videos contain entirely different entity instances;
- **no identity candidate can be correct**, and the validation trace correspondingly
  reports no identity accuracy at all.

P-SA's expected embedding is a broad mixture over 2,474 columns that are all wrong
by construction (measured attention: normalized entropy 0.53–0.65, mean max
probability 0.25–0.32), injected at ~4–5% of the state norm.

**We tested identity feedback in exactly the regime where the paper reports it as
near-null, and we have not tested the regime where the paper reports it as
decisive.** That is a scope limitation of the experiment, not evidence against the
mechanism.

## 6. Two concrete gaps in the corrected experiment

**(a) P-Direct was dropped from both corrected arrays.** The previous round had
`runs/pair-seed0/pair-p-direct-seed0`; the corrected arrays contain only
`integral-none` and `integral-p-sa`. So the corrected experiment currently does
**not measure the mechanism the paper actually claims for binary labels** — the
dynamic context layer. This is the cheapest missing cell and it restores the
paper's own comparison.

For reference, the *old* (pre-correction) run gave P-Direct 42.46 vs Integral
no-feedback 44.16 R@1: +1.70 pp for dynamic context, against the paper's +15.16 pp
on VRD. Worth re-measuring properly on the corrected setup before drawing any
conclusion from that gap.

**(b) Category feedback is off by design.** `fidelity.md:99` — "Category
predictions are not fed back initially." But the paper's *second* stated benefit of
top-down connectivity is exactly label biasing (§6.4):

> Second, labels would get biased: If Sparky is detected in the scene, it will bias
> the unary labeling toward the unary label Black if this is known from semantic
> memory to be true.

And our own complementarity study established that on PVSG, **categories carry
most of the predicate signal** while identity carries none that is usable on
development videos. Category feedback is therefore the top-down pathway most
likely to matter on this dataset, and it is the one we have not run.

This also raises the documented paper/reference-code discrepancy from a fidelity
footnote to a scientific question. The paper restricts P-SA attention to entity
columns (§5.3: "In matrix `A` we only consider the columns relating to entities"),
which is what we implemented — the 2,474 candidates are exactly the identity group.
The authors' reference code appears to score the whole concept/entity bank. On
PVSG, the reference-code variant is the one with a plausible mechanism.

## 7. What would settle the question

In priority order:

1. **Known-identity pair evaluation (VRD-EX analogue).** Evaluate the corrected
   P-SA checkpoint on held-out frames of *training* videos, where the identity bank
   contains the entities being seen. This is the direct transfer of the paper's
   VRD-EX result and the only condition in which identity feedback can carry
   information. The object protocols already have the machinery — blocked
   known-identity recognition and few-shot enrollment (`fidelity.md:116`, `:129`) —
   but the pair experiment does not use it. **If feedback is null here too, that is
   a real negative result about the mechanism. Until then, it is untested.**
2. **Corrected P-Direct**, restoring the paper's actual binary-label comparison
   (dynamic context, not feedback). One job per array.
3. **Category feedback**, i.e. P-SA over category columns or over the full bank, as
   the reference-code variant. This tests the paper's label-biasing benefit on the
   information PVSG actually contains.
4. **Evaluation-time zero-feedback intervention** on the existing P-SA checkpoint —
   separates "feedback is uninformative" from "the model learned to ignore
   feedback". Cheap, still outstanding from two rounds ago.

Steps 1 and 3 are the ones that could turn the current null into a positive result.
Step 2 is what the paper's binary-label claim is actually about. None of them
require changing the architecture.

## 7b. What is in the bank versus what is actually fed back

Measured from `runs/pair-corrected-qtb-seed0/.../vocabulary.json` and
`experiments/pvsg/models.py:321-341`.

**The bank (`A`, 2,783 columns) contains everything:**

| group | columns | global range |
|---|---:|---|
| predicate | 59 | 0–58 |
| object_category/source | 120 | 59–178 |
| object_category/fine | 116 | 59–205 (89 shared with source) |
| object_category/basic | 76 | 206–281 |
| object_category/coarse | 22 | 282–303 |
| object_category/domain | 5 | 304–308 |
| identity | 2,474 | 309–2,782 |

250 distinct category columns, 59 predicate columns, 2,474 identity columns.

**The feedback candidate set is identity-only.** `IntegralTB.forward` calls
`_index_feedback(self.brain, q, identity_candidates, feedback_mode)` at both the
subject and object windows. Category logits are computed *after* feedback
(`_category_logits(self.brain, q, category_candidates)`) but are never fed back;
predicates are never fed back. The trace confirms it: `candidate_count = 2474` at
every attention diagnostic.

This is the paper-faithful choice (§5.3: "In matrix `A` we only consider the
columns relating to entities") and is recorded at `fidelity.md:99` ("Category
predictions are not fed back initially"). It is a configuration decision, not a
limitation of the bank.

## 7c. Identity embeddings *are* semantically grounded — and that is why feedback is constant

The "a different dog of the same breed should still be informative" hypothesis was
tested directly on the checkpoints.

**The grounding worked.** Identity columns are strongly aligned with the category
subspace:

| | QTB | original | random null |
|---|---:|---:|---:|
| mean identity↔category cosine | 0.295 | 0.452 | −0.001 |
| mean max-per-identity cosine | 0.608 | 0.726 | 0.091 |
| identity energy in top-10 category PCs | **51.6%** | **70.4%** | 1.3% |
| identity energy in top-50 category PCs | 57.6% | 76.5% | 6.5% |

So the joint category supervision did what it was introduced to do: identity
embeddings live largely inside the category subspace. The mechanism the hypothesis
requires is structurally present.

**But the identity columns are highly collinear:**

| | QTB | original |
|---|---:|---:|
| mean identity↔identity cosine | 0.479 | 0.678 |
| fraction of identity pairs with cosine > 0.3 | 87.0% | 100.0% |

Consequence: any attention-weighted average over them is dominated by a common
component. Simulating 400 attention distributions at the observed perplexity
(161 for QTB, 70 for original — from `entropy_mean` 5.08 / 4.35 nats over 2,474
candidates):

| | QTB | original |
|---|---:|---:|
| cosine between expected-feedback vectors of *different* draws | **0.9946** | **0.9935** |
| ‖common component‖ | 1.740 | 2.568 |
| mean ‖residual‖ | 0.135 | 0.208 |
| **residual / common** | **0.078** | **0.081** |
| observed across-example norm variation (`l2_std/l2_mean`) | 18–22% | 8–9% |

**This is the mechanism of the null.** The injected vector is ~92% the same vector
on every example. Combined with `l2_over_pre_feedback_q ≈ 0.04–0.05`, the
*example-specific* part of the feedback is roughly **0.4% of the state norm**. The
constant part is a fixed bias that the no-feedback model learns directly into `A`,
which is exactly why the two runs converge to functionally identical parameters.

Caveat: the simulation draws attended sets at random, whereas real attention
selects example-relevant identities and could deviate further from the global mean.
The observed across-example norm variation (8–22%) is the same order as the
simulated residual ratio (8%), so the conclusion is consistent with both, but norm
variation is not the same measurement as direction variation. Tracing the
across-example variance of the feedback *direction* would settle it exactly.

## 7c-bis. What the paper says about the feedback candidate set

**Algorithm 1 (p.19) scores the subject and object over the full concept set `C`,
not an entity-only subset:**

```
14  q_T  ← q̃_T + a_{t*}                        episodic index feedback
15  ∀c ∈ C : n_C(c) ← a_c^T sig(q_T)
18  ∀s ∈ C : n_S(s) ← a_s^T sig(q̃_S)           subject scored over C
19  sample s* ~ softmax_β(n_S)
20  q_S  ← q̃_S + a_{s*}                        subject index feedback
21  ∀c ∈ C : n_C(c) ← a_c^T sig(q_S)           unary label scored over the SAME C
22  sample c* ~ softmax_β(n_C)                 ... but c* is never fed back
25  ∀o ∈ C : n_O(o) ← a_o^T sig(q̃_O)           object scored over C
27  q_O  ← q̃_O + a_{o*}                        object index feedback
30  q_P  ← q̃_P                                 NO feedback at the predicate window
```

Line 18 and line 21 use the *same* set `C`. The paper justifies this explicitly
(p.13):

> Distinguishing between entities, classes, and attributes is important for some of
> the discussions. **But generalized statements permit us to treat all concepts
> almost identically in the algorithmic implementation in Section 5.**

Only §5.3, defining the *attention approximation*, narrows it:

> In matrix `A` we only consider the columns relating to entities.

So the entity-only restriction is a property of the P-SA approximation, not of the
underlying algorithm. Widening the subject/object candidate set to all concepts is
the literal reading of Algorithm 1 and matches the authors' reference code. It
should be classified **paper-faithful (Algorithm 1 reading)**, not as an extension.

**Feeding back the decoded category `c*` is a different operation, and it is an
extension on the perception path.** Algorithm 1 samples `c*` at line 22 and never
injects it. But the paper defines exactly that operation elsewhere:

- p.18: "`q̈_S ← q_S + a_{c*}` is the embedding of the sentence `(s*, hA, c*)` in
  the context of the scene" — shown in Figure 2 as additional processing steps.
- §5.2, Embedded Symbolic Reasoning by Chaining: decoded labels feed back and drive
  further decoding (Sparky → Dog → Mammal).
- p.4: "These sequences then give feedback to the representation layer and earlier
  processing layers and thus inform the brain as a whole about what has been
  decoded."

So category feedback is a paper operation used for chaining and semantic
completion; putting it on the perception forward path to the predicate is an
**experimental extension**. The repository already implements the operation as the
evaluation-only sequential hierarchy rollout (`fidelity.md:125`).

**QTB makes the candidate set a first-class variable.** qtb_LATEST Eq. (40):

> `P(k) ← z_k softmax_z( a_{0,k} + Σ_ℓ γ_ℓ a_{ℓ,k} )`
> where `softmax_z` denotes the softmax restricted to the postselected outcomes.
> **This is the update used in the TB.**

The indicator `z ∈ {0,1}^N` is causal postselection (§8.2). Entity-only is one
choice of `z`; the full concept set is another. §12.2.3 notes that causal
postselection *introduces an order effect* that global postselection does not — so
`z` is a consequential named variable in the newer theory, not an implementation
detail.

### Would widening hurt any paper hypothesis?

| hypothesis | effect |
|---|---|
| Novel-entity labelling (§5.3: `s0` "does not need to be identified as a stored entity `s*`") | **Helped.** With class columns available, attention can put mass on classes when the entity is novel — closer to the stated claim, not further. |
| Episodic/semantic memory formation via bidirectional `A` | Unaffected. Episodic feedback (line 14) is a separate candidate set. |
| One-brain hypothesis (§8.6) | Unaffected. Still one representation layer, still sequential. |
| "The brain is a sampling engine: only activated indices are communicated" | Mildly loosened. A softmax over 2,724 candidates is a looser approximation to sampling than over 2,474. Mitigate by reporting per-group attention entropy (already traced) and by preferring P-Samp/winner feedback for categories. |
| Order effects under causal postselection (QTB §12.2.3) | This *becomes testable* rather than being harmed. |

Two real risks, both avoidable by preserving Algorithm 1's ordering:

1. **Self-confirmation.** Never re-score a group after feeding back its own decoded
   index. Algorithm 1's order is score → sample → feed back → score the *next*
   group. With five hierarchy levels this needs an explicit level order, which the
   sequential hierarchy rollout already defines.
2. **The predicate window.** Line 30 is `q_P ← q̃_P` — no feedback there. Feeding
   back a predicted predicate before scoring the predicate would be circular.

A third, non-architectural risk: if the *true* category is ever teacher-forced into
the feedback, the model becomes the oracle-category baseline (60.07% R@1 in the
complementarity study) and must be reported as an oracle ceiling, not as TB
performance.

### Will widening actually fix the constant-feedback problem?

Not by dilution. Category columns are much less collinear than identity columns,
but they are outnumbered 10:1, so a *uniform* mixture over the union is still
dominated by identities:

| candidate set (QTB / ORIG) | n | mean pairwise cosine | residual/common under uniform attention |
|---|---:|---:|---:|
| identity | 2,474 | 0.479 / 0.678 | 0.100 / 0.067 |
| category | 250 | **0.185 / 0.317** | **0.155 / 0.116** |
| concepts (identity + category) | 2,724 | 0.445 / 0.638 | 0.107 / 0.073 |

The fix therefore depends on attention *concentrating* on category columns, not on
their mere presence. The reason to expect it can: the model already predicts
subject/object categories at 78% / 70%, so the category winner is genuinely
recognizable, unlike the identity winner which cannot be correct at all on
held-out videos.

This makes a concrete prediction: **P-Samp (winner-take-all) over category
candidates should be the strongest condition**, because it injects one specific
category embedding rather than a mixture that re-averages toward the common
direction. That mirrors the paper's VRD-EX result, where P-Samp was best precisely
when the index was recognizable. A soft P-SA over all 2,724 concepts at the
observed entropy (~5 nats) risks reproducing the same washing-out we measured.

## 7d. The three protocols and their VRD analogues

From `experiments/pvsg/protocols.py` and `experiments/pvsg/materialize.py:50-65`:

| protocol | construction | entities at evaluation | VRD analogue |
|---|---|---|---|
| `heldout_video/` | 15% of official-training videos reserved by salted hash (`fidelity.md:128`) | **entirely novel** — different videos | **VRD-E** |
| `blocked/` | within each video: 45% observation / 10% embargo / 45% evaluation (`blocked_boundary`) | **same instances**, later frames | **VRD-EX** |
| `fewshot/` | earliest 10 mask-visible supports ≥5 frames apart, 25-frame query embargo | enrolled identities, k ∈ {1,3,5,10} | no analogue — new capability |

Both `blocked/train_pairs.jsonl` and `blocked/evaluation_pairs.jsonl` **already
exist** in the materialized manifests. `pair_experiment.py:623-626` hardcodes
`heldout_video/`, so the blocked pair protocol needs a path parameter, not a new
experiment.

Note the blocked protocol is a *better* VRD-EX than VRD-EX: the paper's version
distorted copies of the training images (p.23), so "known entity" partly meant
"nearly the same pixels". Blocked uses genuinely later frames of the same video
with a temporal embargo, which is real re-identification rather than image
memorization.

Few-shot enrollment is a third, separate thing and not the VRD-EX analogue.

## 8. How the null should be stated

Not: "index feedback does not help."

But: *"With a semantically grounded pair vocabulary, identity feedback over a
training-identity bank has no measurable effect on predicate prediction for novel
entities, on either evolution block, with either expected or sampled injection
(+8 and +3 assignments out of 67,134, sign-inconsistent). This matches the paper's
own novel-entity unary result (P-SA +0.12 points over P-Direct on VRD-E). The
regime in which the paper reports a large feedback benefit — known entities,
VRD-EX, +6.37 points — has not yet been tested here, and neither has category
feedback."*
