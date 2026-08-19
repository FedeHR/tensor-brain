"""What type is ans_correct, really?"""

from collections import Counter

from datasets import load_dataset

ds = load_dataset("hkust-nlp/dart-math-pool-math", split="train").select_columns(
    ["query", "ans_correct"]
)
vals = Counter()
for i, row in enumerate(ds):
    if i >= 20000:
        break
    v = row["ans_correct"]
    vals[(type(v).__name__, repr(v)[:20])] += 1
for k, n in vals.most_common(10):
    print(f"  {k}  x{n}")
