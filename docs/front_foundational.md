# Foundational front: where the Heisenberg update already lives, and what it answers

Scoping document, 2026-08-18. Companion to `docs/BRIEF.md`.
Target: locate the Heisenberg update *precisely* in the classical
statistical/ML literature, so the thesis can claim exactly the part that is
unoccupied — and identify the older question it actually resolves.

Every citation below was surfaced by search during this pass. Bibliographic
details (author/year/venue) were checked against publisher or repository pages;
**the internal content of most papers was not read in full**, so per §3.5 of
`heisenberg_experiment_design.md` treat page-level claims as "verified enough to
cite, not verified enough to paraphrase in a footnote".

---

## 0. The one-paragraph answer

The Heisenberg update is **not** an approximation that happens to be exact under
a contrived gate. It is the *unconditional* likelihood of a counting process,
and the thing the repo has been calling "exact Bayes" is a **conditional**
likelihood — the multinomial obtained by conditioning that counting process on
its own total. The dropped term `M log Z(x)` is precisely the information about
`x` carried by *how many* symbols were emitted. So:

> **`½M² Var[log Z]` is not approximation error. It is the conditionality loss
> of the softmax likelihood** — the Fisher-information analogue of what Efron
> (1977) computed for Cox's partial likelihood.

This is a sign flip on the incumbent framing, it is cheap to state, and it is
hard to dismiss: the additive rule uses *more* of the data than the "exact"
comparator, not less. The whole five-result package then reorganizes into one
axis — **what you condition on** — with claims 3, 4 and 5 of the brief as its
three columns.

---

## 1. Thread 1 — the multinomial–Poisson transformation

### Verdict: **partially known — and the brief's mapping is subtly wrong in a way that matters.**

**What is established.** Conditional on their sum, independent Poisson counts are
multinomial. The transformation is a standard estimation device: Birch (1963),
Palmgren (1981), Baker (1994, *The Statistician* 43:495–504), Lang (1996; 2004
*Ann. Statist.* 32:340–383 on multinomial–Poisson homogeneous models), and the
"Poisson trick" for multinomial logit. Its selling point is that the two branches
give **the same MLE for the emission parameters**, so you may fit a multinomial
model with Poisson software. Baker's paper is explicitly framed as "simplifies
maximum likelihood estimation".

**Where the brief is right.** The taxonomy in §6.1 of
`heisenberg_experiment_design.md` is correct and is the right frame. Fixed top-k
retrieval reinstating the normalizer *is* the multinomial branch, and that is
strong evidence for the mapping — exactly as the brief argues.

**Where the brief is wrong.** "The Heisenberg update is literally the
Poisson-branch posterior" is **not** correct as stated. Write the Poisson branch
out: `n_k ~ Poisson(T e^{s_k(x)}/C)`, log-likelihood

```
sum_k n_k s_k(x)  -  (T/C) Z(x)
```

The normalizer does **not** disappear. It changes shape, from `-N log Z(x)`
(multinomial) to `-(T/C) Z(x)` (Poisson). Both are non-affine in `x`, so **neither
branch gives the additive rule.** The additive rule needs a *third* thing: the
exposure `T` must be **unobserved**, so the compensator has nothing to multiply.

And "unobserved" has to mean genuinely unmodelled, not marginalised. If you put
the scale-invariant prior `pi(lambda) ∝ 1/lambda` on the exposure and integrate,

```
∫ lambda^{M-1} e^{-lambda Z(x)} dlambda  =  Gamma(M) Z(x)^{-M}
```

you get the compensator `-M log Z(x)` back **exactly** — i.e. marginalising the
exposure with Jeffreys' prior *is* conditioning on the count. That is a clean
analytic statement and it sharpens the brief considerably: the Heisenberg regime
is not "exposure unknown", it is "exposure known-and-constant, count informative,
non-events uninstrumented".

**The corrected three-regime table** (supersedes the one in §6.1 by making the
middle row's provenance explicit):

| protocol | compensator | classical home |
|---|---|---|
| reports only; opportunities never instrumented | **none** — additive rule exact | weighted/size-biased sampling; presence-only |
| reports + known exposure `T` | `-(T/C) Z(x)` | Poisson branch; place-cell decoding (Zhang 1998); Poisson factorisation |
| reports + their total count `N` (or exposure marginalised with `1/lambda`) | `-N log Z(x)` | multinomial branch; the paper's model; fixed top-k |

**What is genuinely new.** The MP transformation is stated, universally, as an
equivalence for **inference about the emission parameters**. It is *never* stated
as a statement about the **posterior over a latent state**, where the two
branches are **not** equivalent — they differ by exactly the tilt `Z(x)^M`. Three
sub-claims, all unoccupied as far as this search reached:

1. **Branch choice changes the latent posterior, not just the arithmetic.** The
   MP literature's whole point is that branch choice is free. For a latent state
   it is not free; it is a modelling commitment.
2. **The additive rule is the branch-(a) posterior, and it is conjugate**
   precisely because branch (a) has no compensator.
3. **Exactness ⟺ ancillarity.** The additive rule is exact under the multinomial
   branch iff `log Z` is affine in `T(x)`, i.e. iff `Z` can be gauged constant,
   i.e. iff the total count `N` is **ancillary for `x`**. This is the classical
   justification for conditional inference (Fisher), and the MP literature knows
   that ancillarity of the total is delicate — see the survey *Conditional
   inference of Poisson models and information geometry: an ancillary review*
   (*Information Geometry*, 2022, doi:10.1007/s41884-022-00082-w), which states
   that the conditioning variable is not ancillary in the exact sense except
   under product-multinomial sampling.
   **Nobody has connected that delicacy to the cost of an additive update.**

**Cost of the framing.** It makes the claim protocol-dependent and therefore
falsifiable — good — but it also means the flagship application can never be a
fixed-count corpus. Captions-per-image is fixed at 5 in COCO; the *word counts*
within them vary, so the COCO design survives, but the point has to be made
explicitly or a reviewer will make it for you.

**Citations.** Baker, *JRSS-D* 43(4):495–504 (1994). Lang, *Ann. Statist.*
32(1):340–383 (2004). Birch (1963). *Information Geometry* (2022),
doi:10.1007/s41884-022-00082-w.

---

## 2. Thread 2 — self-normalising / normalizer-free estimation

### Verdict: **already known at training time; the inference-time version is genuinely unoccupied, and the distinction is real.**

**What is established, and it is a lot.**

- **NCE** (Gutmann & Hyvärinen 2010/2012) treats `log Z` as a free parameter and
  estimates it. **Mnih & Teh (2012)** discovered empirically that *fixing* `Z_c`
  to a constant costs nothing for language models — the model self-normalises.
- **Devlin et al. (2014)** added an explicit self-normalisation penalty to the
  training objective for decoding-time speed.
- **Andreas & Klein (2015)**, *When and why are log-linear models
  self-normalizing?* (NAACL) — this is the same object as the repo's §12b
  `Var[log Z]` regulariser, invented for unrelated reasons. Already flagged in
  §3.2 of the design doc; confirmed here.
- **Goldberger & Melamud (2018)**, *Self-Normalization Properties of Language
  Modeling* (COLING) — analyses when NCE-with-fixed-`Z` self-normalises.
- **Levy & Goldberg (2014)** — SGNS implicitly factorises shifted PMI; the shift
  is a constant, i.e. a *global* normalizer absorbed into a bias.

**The distinction the brief proposes is real and is the whole story.** Every one
of the above avoids `Z` while *fitting the emission model*: the target is
`theta`, the data are `(context, symbol)` pairs, and `Z(context)` is a nuisance.
None of them is doing **belief updating over a latent state at inference time**,
where `Z(x)` is not a nuisance at all — it is a *likelihood term about the very
quantity you are inferring*. That inversion (nuisance → signal) is the
unoccupied ground, and it explains an otherwise puzzling asymmetry:

> Self-normalisation research wants `Var[log Z]` small because a fluctuating `Z`
> is expensive. The Heisenberg analysis wants `Var[log Z]` small because a
> fluctuating `Z` is **informative**, and an additive rule cannot hear it.
> Same penalty, opposite reason. One is a compute argument, the other an
> information argument.

That sentence is worth a paragraph in the thesis; it converts the Andreas–Klein
"we got scooped" risk into a contribution (a second, independent justification
for an existing penalty, with a different optimal `lambda` regime — and §12d's
data-limited negative is exactly the sort of thing the compute framing would
never predict).

**Nothing found** stating "selection proportional to `Z` makes the naive rule
exact". Targeted searches on that phrasing returned nothing on point. Energy-based
models, ranking losses and contrastive objectives all sidestep `Z`; none makes it
cancel by a *sampling design*.

**One classical object to cite defensively.** The exponential-race / Gumbel-max
representation (Yellott 1977; Maddison, Tarlow & Minka 2014, *A\* Sampling*;
Maddison 2016, *A Poisson process model for Monte Carlo*, arXiv:1602.05986)
contains the exact structural fact behind the gate: for competing exponentials
with rates `e^{s_k(x)}`, the **identity of the winner** is `softmax(s(x))` and
the **time of the winner** is `Exp(Z(x))`, and the two are **independent**. The
Heisenberg gate `pi(x) = Z(x)/C` is the linearisation of "did the race finish
inside the window", and the cancellation is that independence. Presenting the
gate this way makes it look inevitable rather than contrived, and pre-empts
"why *that* gate?".

---

## 3. Thread 3 — ignorability and selection

### Verdict on the case-control analogy: **partially known, and the correct statement is better than the one the brief guessed.**

The brief asks whether the affine gauge of `log Z` is the same object as the
Manski–Lerman intercept shift. **It is not — it is its transpose,** and saying so
precisely is the strongest single result in this document.

Write the score matrix `S[k, x] = a_{0,k} + a_k^T x`. The multinomial likelihood
sees `S` only through differences across `k`. So there are two "blind" directions:

| direction | form | status under the multinomial branch | status under the Poisson branch |
|---|---|---|---|
| **column-constant** (varies with `k`, same for all `x`) | `S -> S + g(k)` | **identified**, but confounded by outcome-based sampling | identified |
| **row-constant** (varies with `x`, same for all `k`) — *the gauge* | `S -> S + f(x)`, i.e. `A -> A - c1^T` | **not identified at all** | **identified** (it is the total rate) |

- The **column-constant** direction is the case-control / choice-based-sampling
  object: Manski & Lerman (1977, *Econometrica* 45(8):1977–88); Prentice & Pyke
  (1979, *Biometrika* 66:403); Scott & Wild (1986, *JRSS-B* 48(2):170). Sampling
  on the outcome shifts only alternative-specific constants; slopes are
  consistent. Its modern ML incarnation is the **logQ correction** in two-tower
  retrieval (Yi et al., RecSys 2019; and *Correcting the LogQ Correction*,
  RecSys 2025, arXiv:2507.09331) — literally the same intercept shift,
  rediscovered.
- The **row-constant** direction is the Heisenberg gauge. In random-utility
  language it is the textbook fact that *only utility differences are
  identified* (McFadden). It is invisible to every choice-data method ever
  written **and the Heisenberg update depends on it.**

So the citable reframing is:

> **The Heisenberg update is sensitive to the one direction in the readout that
> choice data cannot see.** The gauge is the transpose of the case-control
> intercept: case-control is corrected by an outcome-side constant, the gauge by
> a state-side constant. And the gauge is exactly the parameter the *Poisson*
> branch identifies and the *multinomial* branch does not — which is why
> gauge-fixing is free (it costs no fit) and yet changes beliefs by 53%.

That last sentence is a *derivation* of the repo's most surprising empirical
result (§12a: 53% of belief error removed at zero fit cost), from identification
theory alone. It was previously reported as a lucky fact. It is not luck: fit
cannot change, because the gauge is unidentified from the fit; belief must
change, because the belief update reads a parameter the fit never constrained.

**Standing warning, and it is the same warning.** Fithian & Hastie (2013,
*Ann. Appl. Statist.* 7(4):1917–1939, *Finite-sample equivalence in statistical
models for presence-only data*) state that in the presence-only IPP model the
intercept "reflects nothing more than the total number of presence samples". That
is regime (a)'s identification limit, and it transfers verbatim: regime (a)
identifies *relative* state likelihood, not absolute rate, because `C` and `T`
are confounded. Cite Warton & Shepherd (2010) and Fithian & Hastie (2013) before
a reviewer does. Presence-only SDM has been in regime (a) for fifteen years.

### Verdict on "ignorability by cancellation": **partially known — the general condition exists; this instance does not.**

- **Rubin (1976)**, *Inference and missing data*, *Biometrika* 63(3):581 — MAR +
  parameter distinctness are "the weakest general conditions under which ignoring
  the process **always** leads to correct inferences". The word *always* is the
  opening: they are sufficient, not necessary.
- **Dawid & Dickey (1977)**, *Likelihood and Bayesian inference from selectively
  reported data*, *JASA* 72(360):845–850 — the direct precursor. They ask when
  the face-value likelihood needs no modification **and derive conditions under
  which none is necessary**. Any claim of "ignorability by cancellation" must
  cite this and position against it.
- **Molenberghs, Beunckens, Sotto & Kenward (2008)**, *JRSS-B* 70(2):371–388 —
  every MNAR model has an MAR counterpart with equal fit. This is a *threat*:
  a reviewer will say the Heisenberg gate is "just" an MNAR model with an MAR
  twin. The answer is that the twin has a different *latent* posterior, which is
  exactly the point of §1 above, and that the two are separable from symbol
  frequencies alone (§7 of `tb_update_generalized.md`).
- **Rao (1965); Patil & Rao (1978); Bayarri & DeGroot (1992)**, *A "BAD" view of
  weighted distributions and selection models*, in *Bayesian Statistics 4*
  (Oxford, pp. 17–33) — weighted-distribution theory. The Heisenberg gate is a
  weighted distribution with weight `w(x) = Z(x)`.

**The unoccupied claim, stated tightly:**

> A selection mechanism that depends on the **latent** state is nonignorable in
> Rubin's sense. But when the weight function is the emission model's **own
> normalizer**, `w(x) = Z(x)`, the selection is *self-cancelling*: no correction
> is needed, and applying the principled correction makes beliefs strictly
> worse. This is a constructive instance of Dawid & Dickey's condition in which
> the weight is not chosen for convenience but is **forced by the model** — the
> emission model determines its own ignorable selection rule.

The "self-referential weight" is the part nobody has. Rao/Patil/Bayarri–DeGroot
take `w` as given and exogenous; here `w` is a *derived* quantity of the
likelihood you already wrote down. That also explains why it is the unique
cancelling gate (§1.2 of the design doc: gates must be linear in `Z`), which is
a uniqueness result the weighted-distribution literature has no reason to state.

---

## 4. Thread 4 — event-triggered estimation with negative information

### Verdict: **substantially occupied. The delta is real but narrow — and it is the *opposite* of what the brief guessed.**

This is the thread that came back worst, and it is better to know now.

**What exists.**

- **Sijs, Noack & Hanebeck (2013)**, *Event-based state estimation with negative
  information*, FUSION 2013 (Istanbul); **Sijs & Lazar (2012)**, *IEEE TAC*
  57(10):2650–2655. Non-transmission is treated as an observation and its
  likelihood computed properly. The repo's `c_s` drift is, as §3.2 already
  conceded, a crude version.
- **Han, Mo, Wu, Weerakkody, Sinopoli & Shi (2015)**, *Stochastic event-triggered
  sensor schedule for remote state estimation*, *IEEE TAC* 60(10):2661–2675
  (arXiv:1402.0599). **This is the closest prior art to the brief's claim 3 and
  it is close.** They *design* a randomised trigger so that exact inference stays
  closed-form: the sensor transmits when `zeta_k > exp(-z_k' Z z_k)` with
  `z_k = y_k - ŷ_{k|k-1}` (closed loop; the open-loop version uses `y_k` itself).
  Because the *no-transmission* probability is Gaussian-shaped in the innovation,
  the silence likelihood is **conjugate**, and the exact MMSE estimator remains a
  closed-form Kalman-type recursion with no approximation. The deterministic
  (send-on-delta) trigger destroys Gaussianity; the stochastic one preserves it
  *by construction*.
- Follow-ups: packet-drop extension (arXiv:1810.03310, where the Gaussian form
  breaks and a mixture appears); and — directly on the brief's regime (a) —
  *Stochastic event-triggered remote state estimation over Gaussian channels
  **without knowing triggering decisions**: a Bayesian inference approach*,
  *Automatica* (March 2023), which handles precisely the case where you cannot
  tell a non-transmission from a channel loss.
- **Koch (2007)**, "negative information" in tracking and sensor data fusion.

**So the general idea — "choose the gate so that cheap inference is exact" — is
fully occupied, in the Gaussian branch, since 2015.** The brief's claim 3 is
structurally the categorical analogue of Han et al. That must be stated up front
or the chapter reads as ignorant of a decade of control theory.

**The delta, and it is a genuine one.** The two constructions are **dual**, and
the duality is sharp:

| | positive channel (event fired) | negative channel (silence) |
|---|---|---|
| **Han et al. 2015** (Gaussian; trigger is a function of the *measurement*) | trivially conjugate (plain measurement) | **conjugate by design** — trigger shaped to make it so |
| **Heisenberg gate** (categorical; trigger is a function of the *latent's normalizer* `Z(x)`) | **conjugate by cancellation** — the gate deletes the emission normalizer | **not conjugate**; `log(1 - Z(x)/C)` is not affine |

Han et al. engineer the *silence* to be free. The `Z`-gate makes the *report*
free and leaves silence expensive. Nobody, in either literature, has the
statement that in an exponential family with a softmax readout the **positive
channel needs no correction at all while the negative channel needs exactly one
constant vector** — because nobody has been in the categorical branch. The
`grad log Z(x) = A pi(x)` identity (§6.2 of the design doc) is what makes
"exactly one constant vector" computable in closed form.

**Recommendation.** Do **not** pitch claim 4 as "silence is evidence" — that
framing is 2007–2015 prior art and will be recognised as such. Pitch it as the
**asymmetry**: which channel is free is determined by *what the trigger is a
function of* (measurement vs. latent normalizer), and the categorical branch is
the one where the asymmetry appears. Narrow, correct, defensible; not a chapter
on its own.

---

## 5. Thread 5 — cognitive science and the describability tilt

### Verdict: **genuinely unoccupied as a quantitative model — and the derivation is real, but weaker than the brief hopes.**

**Is there an existing model of the form `P_true(x) · Z(x)^M`? No.** The nearest
neighbours, and why each is different:

- **Feldman, Griffiths & Morgan (2009)**, *Psychological Review* 116(4):752–782,
  *The influence of categories on perception: explaining the perceptual magnet
  effect as optimal statistical inference*. **The** rational model of categorical
  perception. But its bias is *shrinkage toward category means* driven by
  perceptual noise; it has no term resembling a global nameability normalizer,
  and no dependence on how much the observer said.
- **Huttenlocher, Hedges & Vevea (2000)** — category-adjusted estimation, same
  shape (prototype attraction), same absence.
- **Hatano, Ueno, Kitagami & Kawaguchi (2015)**, *PLOS ONE* 10(6):e0127618, *Why
  verbalization of non-verbal memory reduces recognition accuracy: a
  computational approach to verbal overshadowing*. The only computational model
  of verbal overshadowing found. It is a **PDP/connectionist** model
  (4200-unit retinotopic input, 20 hidden units, 4200 visual + 6 verbal output
  units, LENS, backprop-through-time) instantiating the *recoding interference*
  hypothesis. It is mechanistic, not normative — and it offers **no rational
  justification for why verbalisation should distort at all**. Its Simulation 5
  grades description *accuracy* (0/33/67/100%), not description *amount*.
- **Lupyan (2012)**, *Linguistically modulated perception and cognition: the
  label-feedback hypothesis*, *Front. Psychol.* 3:54 — qualitative; labels feed
  back and warp representations toward category-typical values. Correct in
  direction, no equations.
- **Schooler & Engstler-Schooler (1990)**, *Cognitive Psychology* 22:36–71;
  meta-analysis **Meissner & Brigham (2001)**, *Appl. Cogn. Psychol.*
  15:603–616 (29 comparisons, N = 2018, Fisher's Zr = −0.12); Registered
  Replication **Alogna et al. (2014)**, *Perspectives on Psychological Science*.

**Is the derivation real?** Partly. Honest assessment:

*What the derivation genuinely gives.* `Q_H(x) ∝ P(x|k_{1:M}) · Z(x)^M`, exactly,
not approximately. `Z(x)` is the total drive the state delivers to the naming
layer — an operational definition of nameability. So an agent that (i) carries a
factorised state, (ii) updates it additively on emitted labels, and (iii) never
corrects for the normalizer, will hold a memory **biased toward easily-named
states, with the bias growing in the number of labels emitted.** That is verbal
overshadowing, with a dose-response, from three assumptions none of which mention
memory distortion. That is a real derivation and it is not circular.

*What it does not give, and the chapter must say so.* It derives the effect from
an *irrational* agent — one using the wrong (unconditional) measurement model.
Under the gated protocol the same agent is exactly Bayesian and shows **no**
tilt. So the correct headline is **not** "verbal overshadowing is rational". It is:

> **Verbal overshadowing is the signature of a specific measurement-model
> mismatch.** An additive, order-invariant, normalizer-free belief updater is
> exactly Bayesian when its own speech acts are saliency-gated, and *only then*.
> Experimenter-induced verbalisation is ungated — it forces reports the gate
> would not have opened — so the same machinery that is exact in the wild is
> biased in the lab, by exactly `Z(x)^M`.

**That is a much better claim than "we derived verbal overshadowing".** It
predicts the paradigm's own weirdness: the effect is famously fragile,
context-dependent and instruction-sensitive (Meissner & Brigham's moderators are
crime–description delay, description–lineup delay, instructions, and *number of
repetitions of the verbal description*), which is exactly what a mismatch account
predicts and a pure interference account does not.

**Is there a dataset that could test the `M` / `M²` dose-dependence?** Yes,
weakly, and the literature already points at it:

- Meissner & Brigham (2001) list **number of repetitions of the verbal
  description** as a moderator — that is `M` directly, and the meta-analytic
  data already exist.
- **Meissner, Brigham & Kelley (2001)**, *Memory & Cognition* 29(1):176–186,
  and the "instructional bias" line: elaborative / "report everything"
  instructions produce reliably larger overshadowing than free description.
  Forced elaboration = larger `M` = larger tilt. **Sign matches.**
- The RRR (Alogna et al. 2014) fixed verbalisation at a single 5-minute dose, so
  it cannot test dose-response — which is itself an argument for running it.

**The falsifiable prediction the tilt makes and no incumbent account makes:**
the tilt is `Z(x)^M`, so the *log*-odds distortion is **linear in `M`** while the
KL from the true memory is **quadratic in `M`**. A recognition-memory experiment
with 1 / 2 / 4 / 8 successive descriptions of the same face should show a
**d′ decrement that accumulates linearly in the number of descriptions**, not a
saturating "one description does it all" curve — which is what recoding
interference (one corrupted trace, overwritten once) predicts.
**Interference predicts saturation; the tilt predicts linear accumulation.**
That is a clean, cheap, opposite-sign contrast, on an existing paradigm, with an
existing meta-analytic moderator to power it.

Second prediction, sharper and cheaper: the tilt is toward **high-`Z`** states,
i.e. states with high *total* drive across the whole vocabulary — not toward the
nearest prototype. Feldman–Griffiths–Morgan predicts attraction to the **nearest
category mean**; the tilt predicts attraction toward **globally nameable**
regions, which for a stimulus sitting between two categories is a *different
direction*. Stimuli equidistant from two prototypes separate the two models.

---

## 6. Thread 6 — other foundational neighbours

Brief verdicts, all "known", listed so the thesis can cite defensively.

- **Bayesian brain / predictive processing / FEP.** The additive update is a
  variational message in a mean-field scheme; nothing new. But the *gating*
  result has a genuine FEP-flavoured reading that is not in that literature: an
  agent can make its own cheap inference rule exact by controlling **when it
  samples**. That is a niche-construction argument about the observation process,
  distinct from the usual "act to minimise surprise". Worth one paragraph, not a
  chapter. Nothing found occupying it (searches on resource-rational sampling
  policies that make an approximation exact returned only AIAS-style
  active-sampling work, which is about *which computation to run*, not about
  making a rule exact).
- **Product of experts (Hinton 2002) / exponential-family harmoniums (Welling,
  Rosen-Zvi & Hinton 2004).** The normalizer problem there is about *learning*
  (hence contrastive divergence). The Heisenberg question is about *belief
  updating* with the model fixed. Same object, different use; cite, do not build
  on.
- **Modern Hopfield / associative memory (Ramsauer et al. 2021).** Already
  dropped in §3.4 of the design doc as a reframing. Concur.
- **Khan & Rue (2023), Boyen & Koller (1998), Zhang et al. (1998).** Already in
  §3.2. The §6.1 taxonomy defuses Zhang correctly (spike counting is regime (b),
  because a time bin is an instrumented exposure). That defence is sound and
  should be in the chapter verbatim.
- **Cox (1972, 1975) partial likelihood, and Efron (1977)**, *JASA*
  72(359):557–565, *The efficiency of Cox's likelihood function for censored
  data*. **The best structural analogue found in this whole pass, and it is not
  in §3.2.** Cox's partial likelihood has a risk-set denominator
  `sum_{j in R} exp(beta' z_j)` — a softmax normalizer — obtained by
  **conditioning on the event times**, i.e. by discarding the information in
  *when* and *how often* events occurred. Efron computed exactly how much
  information that costs, and found the loss small under conditions likely to
  hold in practice. Map: risk set ↔ vocabulary; baseline hazard ↔ the gauge;
  partial likelihood ↔ the multinomial branch; full intensity likelihood ↔
  regime (b); "events observed, at-risk time unrecorded" ↔ regime (a).
  **`Var[log Z]` is the Efron information-loss calculation for a softmax readout
  of a latent state.** Cite this and the chapter acquires a 50-year pedigree.

---

## 7. The single most defensible new claim

Everything above collapses to one sentence, and it should be the chapter's
thesis statement:

> **The softmax likelihood of a latent-state readout is a *conditional*
> likelihood — it conditions on the number of symbols emitted. The additive
> (Heisenberg) update is the corresponding *unconditional* posterior. The two
> agree exactly iff the emitted count is ancillary for the state, equivalently
> iff `log Z` is affine in the carried statistic in some gauge; otherwise they
> differ by the exponential tilt `Z(x)^M`, whose KL cost `½M² Var[log Z]` is
> precisely the state-information carried by the event count. Which of the two is
> correct is a property of the observation protocol — whether non-events are
> instrumented — not a property of the inference algorithm.**

Four corollaries, each independently defensible, each a sign flip on incumbent
advice:

1. **The additive rule uses more of the data, not less.** The "principled"
   normalizer correction *throws away* the count channel. On a threshold-gated
   corpus it is the worst rule available (already verified, §10).
2. **Gauge-fixing is free in fit and large in belief, necessarily.** Because the
   gauge is unidentified in the multinomial branch and identified in the Poisson
   branch. (Derives the 53% result from identification theory.)
3. **`Var[log Z]` has two independent justifications** — compute
   (self-normalisation, Andreas & Klein 2015) and information (this work) — and
   they predict different optimal penalty strengths.
4. **Instrumenting non-events is a modelling decision with a price tag.** Log the
   opportunities and you move from regime (a) to regime (b) and *must* pay a
   compensator. There is a real sense in which not measuring is cheaper than
   measuring and then correcting.

---

## 8. Ranking threads 3, 4, 5 as thesis chapters

| | thread 3 (selection / gauge) | thread 4 (event-triggered) | thread 5 (cognitive tilt) |
|---|---|---|---|
| novelty after this pass | **high** — the transpose result and the ancillarity criterion are unoccupied | low–moderate — Han et al. 2015 owns the idea; only the asymmetry is left | **high** as a quantitative model; moderate as a phenomenon |
| citability / pedigree | **excellent** — Rubin, Dawid–Dickey, Manski–Lerman, Prentice–Pyke, Fithian–Hastie, Cox/Efron | good but crowded | thin; the target literature has no equations |
| testability with existing data | **strong** — COCO two-channel design already built; Open Bandit | weak — needs a simulator, no benchmark community | moderate — meta-analytic moderators exist; a new experiment is a psych study, not a thesis chapter |
| risk of "we already knew that" | low, if pitched at the *latent posterior*, never at the MP transformation | **high** | low |
| reviewer surprise | **high** — "your exact-Bayes baseline is the approximation" | low | high, but in a different field's currency |

**Ranking: 3 ≫ 5 > 4.**

**① Thread 3 is the chapter.** It has the sign flip, the pedigree, the built
experiment, and a derivation of the repo's most surprising number. It also
absorbs threads 1 and 6 as its machinery (the MP taxonomy is its §1; Cox/Efron
its framing device). Title it something like *Ignorability by cancellation:
additive belief updating as unconditional inference*. The COCO
caption-vs-instance design (§2.6, already run) is its empirical core, with the
τ estimator (§1.1) as the diagnostic. Nothing here needs new compute.

**② Thread 5 is the best *section* of that chapter, not a chapter of its own.**
The derivation is real and the mismatch framing is strong, but the deliverable is
a prediction for someone else's paradigm and the thesis cannot run the
experiment. Two pages, one figure (`Z(x)^M` tilt vs. `M`), the
linear-versus-saturating contrast against recoding interference, and an explicit
"this is a prediction, not a result". Do **not** claim to have derived verbal
overshadowing; claim to have derived the condition under which it should and
should not appear. If it must be bigger, the equidistant-stimuli prediction
against Feldman–Griffiths–Morgan is the one worth designing properly.

**③ Thread 4 is a subsection and a defence.** One page: state Han et al. 2015
first, then the duality table of §4, then `c_s` as the categorical-branch
constant. Its job is to stop a reviewer from saying "this is event-triggered
estimation" — which they will, and which is 80% true.

---

## 9. Open items this pass did not close

- **Not verified in full text:** Dawid & Dickey's exact no-modification
  condition. If it already covers weight functions derived from the likelihood
  itself, the novelty of §3 shrinks to the *specific* weight `w = Z`. Read it
  before writing the chapter — it is six pages.
- **Not verified in full text:** Baker (1994) — check whether he anywhere
  discusses the latent-variable case. Believed not, but §1's claim rests on it.
- **Not checked:** whether doi:10.1007/s41884-022-00082-w states an ancillarity
  condition equivalent to "`log Z` affine". If it does, §1's sub-claim 3 becomes
  a citation rather than a contribution. Springer paywalled this pass.
- **Not searched:** the psychophysics literature on *codability* effects in
  colour memory. An APA paper on prototypical bias in hue recognition and the
  role of labelling (*JEP:LMC*, xlm0000357) looks directly relevant to §5's
  second prediction and was not read.
- **Not resolved:** whether the `M²` dose-response is testable on the *existing*
  Meissner & Brigham moderator data, or needs new collection. Worth one hour.
- **Not attempted:** a numerical check that the exposure-marginalisation identity
  in §1 (`Jeffreys prior on lambda` ⇒ multinomial branch) matches the repo's
  `tau` family. It is analytic and exact, but a three-line confirmation in
  `gate_family.py` would be cheap insurance.
