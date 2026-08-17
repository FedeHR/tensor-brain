# Stage 0: is the Heisenberg update useful for diffusion decoding?

A kill-switch measurement, run before anything is built.

## The idea being tested

Masked diffusion language models generate by unmasking several positions per
forward pass. The obstacle is *joint inconsistency*: tokens chosen independently
from their marginals form incompatible configurations — the standard example is
"New City" instead of "New York". The current state of the art
([Zoabi, Ali, Ringel & Wolf, arXiv:2606.15805](https://arxiv.org/abs/2606.15805))
handles this by choosing *which* positions are safe to commit together, then
committing and paying for a fresh forward pass.

What it never does — verified in the paper's text — is **update the still-masked
positions to account for what was just committed**. Its pairwise interaction
matrix is also built from marginals alone, which the authors concede "does not
explicitly model higher-order structure or recover the true joint conditional
distribution."

That gap is exactly the shape of the Heisenberg update. The exact correction to a
remaining position is a pointwise mutual information vector,

```
log P(x_j = v | ctx, x_i = k)  =  log P(x_j = v | ctx)  +  PMI(x_j = v ; x_i = k)
```

and approximating it by a vector that depends only on the committed token,

```
logits_j  ←  logits_j + a_k        for every still-masked j
```

**is** `q ← q + a_k`, with `q` the logit vector at a masked position. If that
approximation holds, tokens can be committed *and corrected* inside a single
forward pass, which is a throughput win the incumbent cannot get.

The assumption is strong: it says the effect of committing a token is the same
regardless of context, and language dependence is famously contextual. So the
first thing to do is measure it, cheaply, on a small model.

## What stage 0 measures

Per commit event:

1. run the model on a partially decoded sequence → `log π_j` for masked `j`;
2. commit the argmax token `k` at the most confident **content** position `i`;
3. run the model again → `log π'_j`;
4. the ground truth correction is `Δ_j = log π'_j − log π_j`.

The question is what fraction of that correction an additive, state-independent
rule recovers:

```
captured  =  1 − KL(π'_j ‖ softmax(log π_j + â_k)) / KL(π'_j ‖ log π_j)
```

The denominator is what current decoders lose by doing nothing within a pass.
`captured = 0` means the additive rule is worthless; `captured = 1` means it
reproduces a forward pass for free.

### Rules compared

| rule | correction | fitted from |
|---|---|---|
| do nothing | `0` | — the current behaviour within a pass |
| additive, gain 1 | `â_k` | mean `Δ` over other contexts where `k` was committed |
| additive, global gain | `β·â_k` | plus one scalar `β` shared by every event |
| free | `λ·E Eᵀ e_k` | no fitting at all beyond one scalar `λ` |
| oracle gain | `β*·â_k` | `β*` chosen per event — an upper bound, not a method |

The **free** variant needs no new parameters: with tied embeddings the unembedding
*is* the embedding matrix, so `E Eᵀ e_k` is the column of the Gram matrix for the
committed token — how similar every vocabulary entry is to the one just written.
A full `|V|×|V|` correction table would be ~23 G entries and is not an option, so a
low-rank route is a requirement rather than an elegance.

### Honesty controls

- **Leave-one-out.** `â_k` for a given event is estimated from every *other*
  context in which `k` was committed, so nothing is scored against a rule fitted
  to itself.
- **KL-optimal scaling, not least squares.** Fitting the gain by least squares in
  logit space is not the same as minimising KL and can pick a scale that is worse
  than doing nothing. The gain is searched directly, and the grid includes `0`, so
  a scaled rule can never lose to the baseline by construction.
- **Global versus oracle gain** are reported separately. Only the global one is a
  method; the per-event optimum is an upper bound.
- **Content commits only.** The model pads a block's tail with `<|endoftext|>` at
  very high confidence, so a plain most-confident rule would spend every
  measurement on padding, where there is no interaction to detect. Special tokens
  are excluded as *commits* but still allowed as *targets*.

## The model

[`dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1)
— a 0.6 B masked diffusion LM adapted from Qwen3-0.6B with MDLM, carrying the same
block-wise low-confidence-first decoding loop as LLaDA. Tied embeddings, vocab
151,936, 28 layers. It runs on a laptop, which is the point: stage 0 is meant to
be cheap enough that a negative result costs nothing.

This is a pilot, not the headline. The paper's models are LLaDA-8B and Dream-7B,
and a 0.6 B model on GSM8K may not represent them. What stage 0 establishes is
whether the effect exists at all and whether the harness is right.

Prompts are the 1,319 GSM8K test questions, chat-templated with
`enable_thinking=False`.

## Efficiency note

A naive design re-decodes a fresh prefix for every measurement, costing ~12
forward passes per event. Instead one greedy trajectory is walked per question and
a measurement taken at every commit, so the forward pass computed for the
post-commit state **is** the pre-commit state of the next step: one forward pass
per measurement.

## Running it

```sh
curl -sL -o data/gsm8k/test.jsonl \
  https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl

PYTHONPATH=".:src" uv run --extra coco \
  --with "transformers==4.56.2" --with accelerate --with huggingface_hub \
  python -m experiments.diffusion_heisenberg.run_stage0 \
    --questions data/gsm8k/test.jsonl --num-questions 400
```

`transformers==4.56.2` is required: the repo's custom modeling file expects
`Qwen3DecoderLayer.attention_type`, which the pinned 5.14.1 no longer has. The
loader also snapshots the repo locally and strips an `if __name__ == "__main__"`
demo block containing `import dllm` — a package that is not on PyPI, and which
`transformers` refuses the file over even though the block never executes. Weights
and forward pass are untouched.

## Layout

| file | what it does |
|---|---|
| `probe.py` | trajectory walk, correction accumulators, KL and scale search |
| `run_stage0.py` | CLI; scan → fit → score, writes `stage0.json` |

Tests: `tests/test_diffusion_heisenberg.py` (9, offline, no weights downloaded).
