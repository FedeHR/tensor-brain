"""Model-free effect sizes from the difficulty-level table.

The tau regression against pool mass turned out not to be interpretable -- the
public DART pool retains only accepted responses and is already roughly balanced
across difficulty (mean count ~216 at every level), so pool mass does not proxy
the pass rate. The difficulty-level table needs no regression and is what the
data actually supports.
"""

import json
from pathlib import Path

data = json.loads(Path("output/tau/dart.json").read_text())
levels = data["levels"]
print(f"{'corpus':40s} " + "".join(f"{f'L{v}':>8s}" for v in levels) + f"{'L5/L1':>9s}")
for corpus, cells in data["by_level"].items():
    ratio = cells[-1] / cells[0] if cells[0] else float("nan")
    print(f"{corpus:40s} " + "".join(f"{x:8.1f}" for x in cells) + f"{ratio:9.2f}")
