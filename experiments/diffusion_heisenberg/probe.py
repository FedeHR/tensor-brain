"""Stage 0: how additive is the effect of committing a token in a diffusion LM?

A masked diffusion LM decodes by unmasking several positions at once. Committing
a token at position ``i`` should change what the model believes about the other
masked positions ``j``, but current decoders never apply that change -- they
commit and pay for another forward pass. The exact correction is a pointwise
mutual information vector,

    log P(x_j = v | ctx, x_i = k)  =  log P(x_j = v | ctx)  +  PMI(x_j = v ; x_i = k)

and the Heisenberg update approximates it by a vector that depends only on the
committed token,

    logits_j  <-  logits_j + a_k .

This module measures whether that approximation captures anything. It does not
build a decoder; it asks the prior question, cheaply, and can rule the idea out.

Protocol per sample
    1. run the model on a partially decoded sequence      -> log pi_j for masked j
    2. commit the argmax token k at the most confident i
    3. run the model again                                -> log pi'_j
    4. the ground truth correction is  delta_j = log pi'_j - log pi_j

The additive rule is then scored **leave-one-out**: the mean correction for token
``k`` is estimated from every *other* context in which ``k`` was committed, so no
sample is scored against a rule fitted to itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

DTYPE = torch.float32


@dataclass
class Accumulators:
    """Per-token running sums of the correction vector, over contexts."""

    vocab_size: int
    tokens: list[int]
    total: torch.Tensor = field(init=False)
    count: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.total = torch.zeros(len(self.tokens), self.vocab_size, dtype=DTYPE)
        self.count = torch.zeros(len(self.tokens), dtype=DTYPE)

    def add(self, token: int, delta: torch.Tensor) -> None:
        row = self.index[token]
        self.total[row] += delta.to(DTYPE).cpu()
        self.count[row] += 1

    def leave_one_out(self, token: int, delta: torch.Tensor) -> torch.Tensor | None:
        """Mean correction for ``token`` estimated without this observation."""

        row = self.index[token]
        if self.count[row] < 2:
            return None
        return (self.total[row] - delta.to(DTYPE).cpu()) / (self.count[row] - 1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"vocab_size": self.vocab_size, "tokens": self.tokens,
                    "total": self.total, "count": self.count}, path)

    @classmethod
    def load(cls, path: Path) -> Accumulators:
        payload = torch.load(path, weights_only=True)
        instance = cls(vocab_size=payload["vocab_size"], tokens=list(payload["tokens"]))
        instance.total = payload["total"]
        instance.count = payload["count"]
        return instance


def materialize(name: str, cache: Path) -> Path:
    """Snapshot the repo locally and drop a dead demo import from its modeling file.

    ``modeling_qwen3.py`` carries ``import dllm`` inside an ``if __name__ ==
    "__main__"`` block that never runs on load, but ``transformers`` scans imports
    statically and refuses the file because ``dllm`` is not on PyPI. Removing the
    demo block changes no model behaviour; the weights and the forward pass are
    untouched.
    """

    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(name, local_dir=cache / name.replace("/", "__")))
    for source in local.glob("modeling_*.py"):
        text = source.read_text()
        marker = 'if __name__ == "__main__":'
        if marker in text:
            source.write_text(text.split(marker)[0].rstrip() + "\n")
    return local


def load_model(name: str, device: str, dtype: torch.dtype, cache: Path | None = None):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    source = str(materialize(name, cache)) if cache is not None else name
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForMaskedLM.from_pretrained(source, dtype=dtype, trust_remote_code=True)
    return model.to(device).eval(), tokenizer


def build_prompt(tokenizer, question: str, device: str) -> torch.Tensor:
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    return tokenizer(text, return_tensors="pt").input_ids.to(device)


@torch.no_grad()
def trajectory(
    model,
    x: torch.Tensor,
    mask_id: int,
    block: slice,
    *,
    steps: int,
    max_targets: int,
    exclude: frozenset[int],
):
    """Walk one greedy decode, yielding a measurement at every commit.

    Each step commits the most confident *content* token and then advances, so the
    forward pass computed for the post-commit state is exactly the pre-commit state
    of the next step. That makes the whole trajectory cost one forward pass per
    measurement instead of re-decoding a fresh prefix for each.

    Yields ``(token, position, targets, before, after)``; ``before`` and ``after``
    are log-probability rows over the block.
    """

    before = model(x).logits[0, block].to(DTYPE).log_softmax(-1)
    for _ in range(steps):
        masked = (x[0, block] == mask_id).nonzero().flatten().tolist()
        if len(masked) < 2:
            return

        confidence = torch.full((before.shape[0],), -float("inf"), dtype=DTYPE,
                                device=before.device)
        for j in masked:
            if int(before[j].argmax()) not in exclude:
                confidence[j] = before[j].max()
        if not torch.isfinite(confidence).any():
            return

        commit_at = int(confidence.argmax())
        token = int(before[commit_at].argmax())
        x[0, block.start + commit_at] = token
        after = model(x).logits[0, block].to(DTYPE).log_softmax(-1)

        targets = [j for j in masked if j != commit_at][:max_targets]
        if targets:
            yield token, commit_at, targets, before, after
        before = after



def _kl(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) for log-probability rows, summed over the vocabulary."""

    return (log_p.exp() * (log_p - log_q)).sum(-1)


def scaled_kl_curve(
    reference: torch.Tensor,
    baseline: torch.Tensor,
    direction: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    r"""``KL(reference ‖ softmax(baseline + s·direction))`` for every ``s``.

    With ``baseline`` already normalized (``logsumexp = 0``) the curve is

        KL(s) = KL(0) + logsumexp(baseline + s·direction) − s·⟨p, direction⟩

    so each scale costs one log-sum-exp rather than a full log-softmax and a
    weighted sum, and ``⟨p, direction⟩`` is computed once. Exact, not an
    approximation -- it is the same quantity, rearranged.
    """

    probability = reference.exp()
    base = float(_kl(reference, baseline))
    projection = float(probability @ direction)
    values = torch.empty(len(scales))
    for index, scale in enumerate(scales.tolist()):
        shift = float(torch.logsumexp(baseline + scale * direction, dim=-1))
        values[index] = base + shift - scale * projection
    return values


def best_scaled_kl(
    reference: torch.Tensor,
    baseline: torch.Tensor,
    direction: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[float, float]:
    """Smallest ``KL`` over ``scales``, and the scale that achieves it.

    Fitting the scale by least squares in logit space is not the same as
    minimising KL, and can land on a value that makes the approximation worse
    than doing nothing. Searching KL directly gives the honest best case for this
    direction, and the grid includes ``s = 0`` so it can never lose to the
    baseline.
    """

    curve = scaled_kl_curve(reference, baseline, direction, scales)
    index = int(curve.argmin())
    return float(curve[index]), float(scales[index])


@dataclass
class Report:
    """Aggregated stage-0 result."""

    rows: dict[str, list[float]] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        self.rows.setdefault(name, []).append(value)

    def summary(self) -> dict[str, float]:
        return {k: float(sum(v) / len(v)) for k, v in self.rows.items() if v}


def embedding_matrix(model) -> torch.Tensor:
    """A single float32 copy of the embedding table, held on the CPU.

    Materializing this per call is what makes the free correction expensive: at
    151,936 x 1024 a float32 copy is ~600 MB, and building one for every committed
    token exhausts an accelerator quickly. Build it once, keep it off the device.
    """

    return model.get_input_embeddings().weight.detach().to(DTYPE).cpu()


def free_correction(embedding: torch.Tensor, token: int) -> torch.Tensor:
    """The parameter-free correction: unembedding applied to the token's embedding.

    With tied weights the unembedding *is* the embedding matrix, so this is the
    column of the Gram matrix ``E Eᵀ`` for the committed token -- how similar every
    vocabulary entry is to the one just written. No fitting, no new parameters.
    """

    return embedding @ embedding[token]


def load_questions(path: Path, limit: int) -> list[str]:
    questions: list[str] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            questions.append(record.get("question") or record.get("problem") or "")
            if len(questions) >= limit:
                break
    return [q for q in questions if q]


def special_tokens(tokenizer) -> frozenset[int]:
    """Ids that must not be committed: padding, end of text, chat scaffolding."""

    ids = set(tokenizer.all_special_ids or [])
    for token in ("<|endoftext|>", "<|im_end|>", "<|im_start|>", "<|mask|>"):
        found = tokenizer.convert_tokens_to_ids(token)
        if isinstance(found, int) and found >= 0:
            ids.add(found)
    return frozenset(ids)



def _prepare(prompt: torch.Tensor, mask_id: int, new_tokens: int):
    x = torch.cat(
        [prompt, torch.full((1, new_tokens), mask_id, dtype=torch.long, device=prompt.device)],
        dim=1,
    )
    return x, slice(prompt.shape[1], prompt.shape[1] + new_tokens)
