"""How big are the corpora, before we commit to downloading them."""

import json
import urllib.request

REPOS = [
    "hkust-nlp/dart-math-uniform",
    "hkust-nlp/dart-math-hard",
    "hkust-nlp/dart-math-pool-math",
    "hkust-nlp/dart-math-pool-gsm8k",
    "open-r1/OpenR1-Math-220k",
]

for repo in REPOS:
    url = f"https://huggingface.co/api/datasets/{repo}?blobs=true"
    try:
        with urllib.request.urlopen(url, timeout=60) as fh:
            meta = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"{repo:34s} FAILED {type(exc).__name__}")
        continue
    data_files = [
        s
        for s in meta.get("siblings", [])
        if s["rfilename"].endswith((".parquet", ".jsonl", ".json.gz"))
    ]
    total = sum(s.get("size") or 0 for s in data_files)
    print(f"{repo:34s} {len(data_files):4d} files  {total / 1e9:7.2f} GB")
    for s in sorted(data_files, key=lambda s: -(s.get("size") or 0))[:4]:
        print(f"    {(s.get('size') or 0) / 1e9:7.3f} GB  {s['rfilename']}")
