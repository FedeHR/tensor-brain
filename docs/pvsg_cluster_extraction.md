# PVSG cluster feature extraction

This workflow has one fixed root:

```text
/nfs/data8/harjes/MASTER/
├── tensor-brain/          # Git clone and immutable experiment code
├── data/pvsg/
│   ├── archives/          # Pinned Hugging Face files
│   ├── staging/           # Incomplete extraction is isolated here
│   ├── dataset/           # Canonical videos, nested masks, pvsg.json
│   └── manifests/         # Source provenance, audit, one JSONL row per video
├── features/pvsg/         # One atomic .pt artifact per video
├── runs/pvsg/             # Later checkpoints, results, and TensorBoard events
├── slurm/logs/
├── cache/                 # Hugging Face, Torch, uv, and XDG caches
└── tmp/                   # Per-job temporary files
```

No runtime data or log is written into the repository.

## Why PVSG needs its own preparation step

The pinned [PVSG Hub snapshot](https://huggingface.co/datasets/Jingkang/PVSG/tree/main)
contains `pvsg.json` and six ZIP archives, not a Hugging Face `datasets`/Arrow dataset. Archive
members carry author-cluster prefixes such as `mnt/lustre/.../data/ego4d/...`.

The [official preparation helper](https://github.com/LilyDaytoy/OpenPVSG/blob/main/tools/unzip_and_extract.py)
removes those prefixes with `basename`. That works for uniquely named videos but flattens every
mask to names such as `0000.png`, so masks from different videos overwrite one another. This
also contradicts the [official loader's expected nested paths](https://github.com/LilyDaytoy/OpenPVSG/blob/main/datasets/datasets/pvsg_image.py).

`experiments/pvsg/prepare.py` therefore performs a strict normalization:

```text
.../<source>/videos/<video_id>.(mp4|MP4)
  -> dataset/<source>/videos/<video_id>.mp4

.../<source>/masks/<video_id>/<frame>.png
  -> dataset/<source>/masks/<video_id>/<frame>.png
```

Before writing, it verifies the pinned JSON SHA-256, all six official archive MD5s, safe member
paths, unique destinations, exact video IDs, and contiguous `0000.png ... N-1.png` masks. The
verified snapshot has 400 videos and 149,488 masks. It writes `manifests/videos.jsonl` for the
Slurm array and `manifests/source.json` for provenance and annotation warnings. It deliberately
does not create a second tree of decoded RGB frames; workers decode each 5 FPS video directly
and require its frame count and dimensions to match `pvsg.json` and its masks.

The audit currently reports upstream annotation problems without modifying them: 39 relation
records use predicates absent from the published 57-predicate vocabulary, and 24 intervals
violate ordinary in-video bounds. Relation-span endpoint semantics also remain an explicit
decision for task construction. None of these issues changes visual feature extraction.

## Pinned DINOv3 contract

The worker uses the stable Hugging Face `AutoImageProcessor` and `AutoModel` API from
Transformers 5.14.1 with:

```text
model:    facebook/dinov3-vitb16-pretrain-lvd1689m
revision: 5931719e67bbdb9737e363e781fb0c67687896bc
```

This checkpoint is gated. Accept its terms on the
[model page](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) before cluster 
setup. The full RGB frame is resized without cropping or a square warp to an approximately
448-pixel long edge. Both dimensions are rounded to multiples of the 16-pixel patch size, so
the only aspect-ratio approximation is the small amount required by that patch grid. The
extractor separates one CLS token, four register tokens, and the spatial grid dynamically; it
never assumes a fixed token count.

For each frame it stores the normalized CLS scene feature, exact-mask object features, and
enclosing-union features for every simultaneously visible pair. Every artifact also records
the model revision, processor configuration, original and processed dimensions, package
versions, coordinate convention, pooling rules, and float16 storage dtype.

## One-time setup

After cloning into `/nfs/data8/harjes/MASTER/tensor-brain`, run on a login or data-transfer
node with `uv` available:

```bash
cd /nfs/data8/harjes/MASTER/tensor-brain
source cluster/pvsg/common.sh
uv sync --frozen --extra pvsg
uv run --frozen --no-sync hf auth login
./cluster/pvsg/setup.sh
```

Setting `HF_HOME` before authentication keeps the token and all model files under `MASTER`.
Do not place a token in a script or Slurm environment argument. Setup downloads the pinned
10.8 GB archives, retains them for reproducibility, extracts roughly 10.9 GB of canonical
content, caches the gated model once, and creates the 400-row array manifest.

## Submit extraction

Cluster resource names are intentionally not guessed. Pass the local partition, GPU, CPU,
memory, and time settings to the wrapper. Start with one manifest row:

```bash
cd /nfs/data8/harjes/MASTER/tensor-brain
ARRAY_RANGE=0 DINO_BATCH_SIZE=8 ./cluster/pvsg/submit_extract.sh \
  --partition=<gpu-partition> \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=08:00:00
```

The completion line reports frame, object-observation, pair-observation, and artifact-size
counts. Use `ARRAY_RANGE=0-7` (or another explicit small cohort) to collect the inputs required
by the pre-extraction audit described in the experiment design. After that audit is satisfactory,
submit all rows:

```bash
cd /nfs/data8/harjes/MASTER/tensor-brain
MAX_PARALLEL=8 DINO_BATCH_SIZE=8 ./cluster/pvsg/submit_extract.sh \
  --partition=<gpu-partition> \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=08:00:00
```

The wrapper creates the log directory before `sbatch` (Slurm will not do so), derives the array
bounds from the immutable manifest, and submits one video per task. Workers run with the Hub in
offline mode, write to a same-directory temporary file, and rename only after the complete
artifact validates. Rerunning the array skips compatible completed artifacts and refuses to
overwrite an artifact with a different contract.
