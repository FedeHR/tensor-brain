"""Peek at the schemas of the public corpora the tau instrument needs.

Streaming only -- these are hundreds of thousands of rows and we only need to
know which fields carry the recorded count and the independent pass rate.
"""

from datasets import load_dataset

SPECS = [
    ("open-r1/OpenR1-Math-220k", "all"),
    ("open-r1/OpenR1-Math-220k", "default"),
    ("SynthLabsAI/Big-Math-RL-Verified", None),
    ("hkust-nlp/dart-math-uniform", None),
    ("hkust-nlp/dart-math-hard", None),
    ("hkust-nlp/dart-math-pool-math", None),
]

for name, cfg in SPECS:
    try:
        ds = (
            load_dataset(name, cfg, split="train", streaming=True)
            if cfg
            else load_dataset(name, split="train", streaming=True)
        )
        row = next(iter(ds))
        print(f"\n=== {name} [{cfg}] ===")
        for k, v in row.items():
            desc = type(v).__name__
            if isinstance(v, list):
                desc = f"list[{len(v)}]"
            s = str(v).replace("\n", " ")[:90]
            print(f"  {k:28s} {desc:12s} {s}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n=== {name} [{cfg}] FAILED: {type(exc).__name__}: {str(exc)[:150]}")
