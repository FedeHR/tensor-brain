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

Linux installs the official PyTorch 2.7.1 / torchvision 0.22.1 CUDA 11.8 wheels. This runtime
is compatible with the minor partition's RTX 2080 Ti and 535 driver and remains usable on a
later A100 through NVIDIA driver backward compatibility. DINO inference uses fixed FP16
autocast on both partitions; changing hardware therefore does not silently change the feature
precision. The conservative default batch size is 4 for the 11 GB 2080 Ti. Batch size affects
throughput and memory only, not the stored feature contract. Each artifact also records the
PyTorch CUDA runtime, GPU name, and compute capability for provenance.

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

Setup logs archive MD5 progress and ten extraction checkpoints per archive. Completed archive
groups are reported as `already published; skipping`; an archive appears under `dataset/` only
after all of its files have been written and atomically published.

Setting `HF_HOME` before authentication keeps the token and all model files under `MASTER`.
Do not place a token in a script or Slurm environment argument. Setup downloads the pinned
10.8 GB archives, retains them for reproducibility, extracts roughly 10.9 GB of canonical
content, caches the gated model once, and creates the 400-row array manifest.

## Submit extraction

Cluster resource names are intentionally not guessed. Pass the local partition, GPU, CPU,
memory, and time settings to the wrapper. Start with one manifest row:

```bash
cd /nfs/data8/harjes/MASTER/tensor-brain
ARRAY_RANGE=0 DINO_BATCH_SIZE=4 ./cluster/pvsg/submit_extract.sh \
  --partition=minor \
  --gres=gpu:1 \
  --cpus-per-task=2 \
  --mem=8G \
  --time=00:30:00
```

Two CPUs allow PyAV to decode ahead without making the pilot unnecessarily difficult to
schedule. Eight GiB is a deliberately modest initial host-memory request; the worker keeps one
image batch and the current video's compact float16 output tables in host memory. Thirty minutes
is a pilot limit, not an expected runtime. If it is insufficient, the atomic writer leaves no
partial feature artifact and the task can be resubmitted with a measured larger limit.

The worker samples `nvidia-smi` once per second. Its final log line reports peak observed GPU
memory and utilization; the raw CSV remains in the job-specific directory under `MASTER/tmp`.
This sampled value may miss a sub-second transient, so retain at least 10% GPU-memory headroom
when increasing `DINO_BATCH_SIZE`.

The completion line also reports frame, object-observation, pair-observation, and artifact-size
counts. Use `ARRAY_RANGE=0-7` (or another explicit small cohort) to measure the variation across
videos before selecting resources for the full array. After that audit is satisfactory, submit
all rows with limits derived from the cohort rather than the former blanket 32 GiB/eight-hour
request. For example:

```bash
cd /nfs/data8/harjes/MASTER/tensor-brain
FULL_JOB_MEMORY=12G
FULL_JOB_TIME=01:00:00
MAX_PARALLEL=8 DINO_BATCH_SIZE=4 ./cluster/pvsg/submit_extract.sh \
  --partition=minor \
  --gres=gpu:1 \
  --cpus-per-task=2 \
  --mem="$FULL_JOB_MEMORY" \
  --time="$FULL_JOB_TIME"
```

The two `FULL_JOB_*` values are illustrative starting points, not fixed project defaults;
replace them with the cohort measurements plus headroom.

The wrapper creates the log directory before `sbatch` (Slurm will not do so), derives the array
bounds from the immutable manifest, and submits one video per task. Workers run with the Hub in
offline mode, write to a same-directory temporary file, and rename only after the complete
artifact validates. Rerunning the array skips compatible completed artifacts and refuses to
overwrite an artifact with a different contract.

## Audit the completed snapshot

After the extraction array finishes, run the CPU-only audit:

```bash
cd /nfs/data8/harjes/MASTER/tensor-brain
./cluster/pvsg/audit.sh
```

The command validates all feature-table contracts, visible object/pair completeness, finite
features, and relation-to-feature joins. It separately expands PVSG relations as half-open and
inclusive intervals, reports multi-predicate prevalence for both interpretations, and does not
choose one silently. Results are written under
`/nfs/data8/harjes/MASTER/runs/pvsg/audits/dino-schema-v2/` as `report.json` and a sampled
`gallery.html`. The gallery compares the final half-open frame with the extra inclusive endpoint,
overlaying the subject mask in cyan, object mask in magenta, and stored union box in yellow.

This is a CPU and filesystem task. If cluster policy prohibits sustained work on the login node,
request an ordinary CPU allocation and run the same shell script inside it; no GPU is required.

No graphical software is needed on the cluster. The simplest inspection options are:

1. Open the remote directory with VS Code Remote - SSH and click individual PNGs under
   `gallery/`.
2. Copy the compact audit directory to the local machine:

   ```bash
   scp -r <cluster-login>:/nfs/data8/harjes/MASTER/runs/pvsg/audits/dino-schema-v2 ./pvsg-audit
   ```

3. Serve the gallery privately from the SSH login. In one cluster shell:

   ```bash
   cd /nfs/data8/harjes/MASTER/runs/pvsg/audits/dino-schema-v2
   /nfs/data8/harjes/MASTER/tensor-brain/.venv/bin/python -m http.server 8765 \
     --bind 127.0.0.1
   ```

   In a second terminal on the local machine:

   ```bash
   ssh -N -L 8765:127.0.0.1:8765 <cluster-login>
   ```

   Then open `http://127.0.0.1:8765/gallery.html` locally. Binding to loopback keeps the server
   inaccessible from the public network. None of these options requires sudo.
