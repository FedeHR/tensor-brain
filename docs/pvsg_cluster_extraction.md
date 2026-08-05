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
features, and relation-to-feature joins for the 394 retained videos. It accepts exactly the six
reviewed exclusions, expands inclusive relation spans after clipping to valid frames, and reports
multi-predicate prevalence and incomplete evidence. Results are written under
`/nfs/data8/harjes/MASTER/runs/pvsg/audits/dino-schema-v2/` as `report.json` and a sampled
`gallery.html`. The gallery compares the final half-open frame with the inclusive endpoint,
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

## Materialize the initial experiment records

The initial experiment deliberately excludes the six reviewed source-defective videos recorded
in `experiments/pvsg/exclusions.json`. Materialization still requires valid schema-v2 artifacts
for every other video, so this allowlist cannot hide a new extraction failure.

Run the CPU-only materializer once:

```bash
cd /nfs/data8/harjes/MASTER/tensor-brain
./cluster/pvsg/materialize.sh
```

It atomically writes `/nfs/data8/harjes/MASTER/data/pvsg/manifests/section6-v1/`. The snapshot is
object-first: `canonical/frames.jsonl` contains every retained frame and its ascending visible
object IDs and feature rows, including empty lists, and all mask-visible object observations are
usable without any relation join. `canonical/positive_pairs.jsonl` separately retains complete
and incomplete relation evidence. Only the initial four-input pair protocol files require
complete scene, subject, object, and union evidence. The fixed schedules are:

- `heldout_video`: a deterministic per-source 85/15 split of official-training videos for
  training/development, with official-validation videos untouched for final evaluation;
- `blocked`: first 45% observation, middle 10% embargo, final 45% evaluation;
- `fewshot`: the earliest ten visible observations per novel development or validation identity,
  separated by at least five frames, followed by a 25-frame/five-second embargo. Restricting the
  ranked supports to `k in {1, 3, 5, 10}` keeps the identity pool and queries fixed across `k`.

`ontology.json` records the 64 active predicates, actual tracked identities, and per-split
predicate support for both all annotated positive frames and their complete-evidence subset.
`splits.json` freezes exact video membership and `object_hierarchy.json`
freezes the validated reviewed hierarchy. `provenance.json` records every decision, exclusion,
count, source revision, feature provenance group, file size, and checksum. The command refuses
to replace an existing `section6-v1` directory. `span_issues.json` makes the remaining annotation
damage explicit: in the pinned retained snapshot, 44 spans are clipped at the video boundary and
five lie wholly outside their declared videos, so the latter cannot produce a frame record. Two
retained videos also contain self-relations, accounting for 335 canonical frame records without
a two-object union; these remain auditable but are absent from complete-evidence protocol views.

### Replace the unused pre-experiment snapshot

Because no experiment consumed the earlier `section6-v1`, materialize the revised contract to a
candidate directory first:

```bash
cd /nfs/data8/harjes/MASTER/tensor-brain
test -z "$(git status --porcelain)"
test ! -e /nfs/data8/harjes/MASTER/data/pvsg/manifests/section6-v1.next
test ! -e /nfs/data8/harjes/MASTER/data/pvsg/manifests/section6-v1.previous
PVSG_SECTION6_MANIFEST_ROOT=/nfs/data8/harjes/MASTER/data/pvsg/manifests/section6-v1.next \
  ./cluster/pvsg/materialize.sh
uv run --frozen --no-sync python - <<'PY'
import json
from pathlib import Path

root = Path("/nfs/data8/harjes/MASTER/data/pvsg/manifests/section6-v1.next")
provenance = json.loads((root / "provenance.json").read_text())
ontology = json.loads((root / "ontology.json").read_text())
counts = provenance["counts"]
for role, prefix in (("development", "development_"), ("evaluation", "")):
    enrollments = counts[f"fewshot/{prefix}enrollment.jsonl"]
    supports = counts[f"fewshot/{prefix}support_objects.jsonl"]
    queries = counts[f"fewshot/{prefix}query_objects.jsonl"]
    assert enrollments > 0 and supports == 10 * enrollments and queries > 0
    print(role, {"identities": enrollments, "supports": supports, "queries": queries})
print("train-supported predicates:", ontology["train_supported_predicates"])
PY
```

Review the two eligible-identity counts and the newly derived train-supported predicate list.
Only if they are suitable, promote the candidate and remove the unused previous snapshot:

```bash
mv /nfs/data8/harjes/MASTER/data/pvsg/manifests/section6-v1 \
  /nfs/data8/harjes/MASTER/data/pvsg/manifests/section6-v1.previous
mv /nfs/data8/harjes/MASTER/data/pvsg/manifests/section6-v1.next \
  /nfs/data8/harjes/MASTER/data/pvsg/manifests/section6-v1
test -f /nfs/data8/harjes/MASTER/data/pvsg/manifests/section6-v1/provenance.json
rm -rf -- /nfs/data8/harjes/MASTER/data/pvsg/manifests/section6-v1.previous
```

Staging first means a failed materialization cannot damage the existing snapshot. The old
snapshot remains recoverable until the candidate has been inspected and promoted successfully.

## Run the tiny-data overfit gate

The first learning job uses one fixed 200-pair batch, so it does not need a full-dataset
`DataLoader` or additional baselines. Give every attempt a unique descriptive name; the runner
refuses to replace an existing run directory. For the primary paper-motivated condition:

```bash
cd /nfs/data8/harjes/MASTER/tensor-brain
PVSG_OVERFIT_RUN_NAME=integral-original-hierarchy-overfit-seed0 \
  ./cluster/pvsg/submit_overfit.sh \
  --partition=minor \
  --gres=gpu:1 \
  --cpus-per-task=2 \
  --mem=8G \
  --time=02:00:00
```

This trains Integral TB with the original recurrent dynamic context, a hidden dimension equal to
the 768-dimensional DINO state, reviewed hierarchy supervision, Adam at `1e-3`, and at most 5,000
updates. It exits nonzero if the exact-target plus `loss <= 0.01` gate is not reached, but still
saves the checkpoint and complete diagnostic result under
`$PVSG_RUN_ROOT/overfit/$PVSG_OVERFIT_RUN_NAME/`. The Slurm log also reports sampled GPU memory
and utilization.

The shell variables `PVSG_OVERFIT_EXAMPLES`, `PVSG_OVERFIT_LEARNING_RATE`,
`PVSG_OVERFIT_MAX_STEPS`, `PVSG_OVERFIT_HIDDEN_DIM`, `PVSG_OVERFIT_MODEL`,
`PVSG_OVERFIT_EVOLUTION`, `PVSG_OVERFIT_SEMANTIC`, and `PVSG_OVERFIT_SEED` expose only the named
scientific and optimization choices already recorded in `config.json`. Do not change several of
them in an unnamed retry. A failed primary run should first be read through its loss, gradient,
CBS, neutral-score, and feedback traces rather than silently converted into a new architecture.

## Rerun the first full object grid with chunked sampling

After the overfit gate passes, submit exactly 12 object-only conditions:

```text
evolution:     original, qtb
score mode:    centered, softplus-bias
learning rate: 1e-4, 3e-4, 1e-3
```

All other choices are fixed in `object_grid.sbatch`: RMS-normalized DINO evidence, current
activation-matched initialization, P-SA training, Adam without weight decay, batch size 128,
seed 0, 10,000 updates, development validation every 1,000 updates, and a deterministic
20,000-observation validation subset. Rows are shuffled within each video, divided into chunks
of at most 1,024 observations, and the chunks are shuffled globally. This reruns the original
whole-video-block grid without overwriting it. Submit the complete array with:

```bash
cd /nfs/data8/harjes/MASTER/tensor-brain
source cluster/pvsg/common.sh
mkdir -p "$SLURM_LOG_ROOT"
sbatch \
  --partition=minor \
  --gres=gpu:1 \
  --cpus-per-task=3 \
  --mem=16G \
  --time=2-00:00:00 \
  --output="$SLURM_LOG_ROOT/%x-%A_%a.out" \
  --error="$SLURM_LOG_ROOT/%x-%A_%a.err" \
  cluster/pvsg/object_grid.sbatch
```

The script defaults to at most 12 concurrent tasks. Pass `--array=0-11%N` to `sbatch` to lower
that limit; this changes scheduling only. Array tasks are assigned as follows:

| Task | Evolution | Score mode | Learning rate |
|---:|---|---|---:|
| 0 | original | centered | `1e-4` |
| 1 | original | centered | `3e-4` |
| 2 | original | centered | `1e-3` |
| 3 | original | softplus-bias | `1e-4` |
| 4 | original | softplus-bias | `3e-4` |
| 5 | original | softplus-bias | `1e-3` |
| 6 | qtb | centered | `1e-4` |
| 7 | qtb | centered | `3e-4` |
| 8 | qtb | centered | `1e-3` |
| 9 | qtb | softplus-bias | `1e-4` |
| 10 | qtb | softplus-bias | `3e-4` |
| 11 | qtb | softplus-bias | `1e-3` |

Each result is written under `$PVSG_RUN_ROOT/object-grid-chunked/` with a run name beginning
`object-chunked-`, leaving `$PVSG_RUN_ROOT/object-grid/` untouched. Every run contains its
configuration, vocabulary,
best checkpoint, train/validation traces, scale diagnostics, and aggregate P-SA and
P-Samp metrics. Semantic evaluation separates observation-micro, supported-class macro,
identity-macro, and video-macro accuracy. Development videos select the checkpoint through hierarchy loss. The final
within-training blocked interval supplies known-identity re-identification metrics; official
evaluation videos remain untouched for later confirmatory experiments.

## Run the seed-0 unary semantic-feedback comparison

The first thesis comparison fixes QTB evolution, `softplus-bias`, Adam at `1e-3`, hierarchy
supervision, and the 1,024-observation chunk sampler. Its five array tasks train the local DINO
linear probe, fused scene/object linear head, P-Direct, Integral without feedback, and Integral
P-SA. P-Samp and the fine-to-domain sequential hierarchy rollout reuse the P-SA checkpoint.

```bash
cd /nfs/data8/harjes/MASTER/tensor-brain
source cluster/pvsg/common.sh
mkdir -p "$SLURM_LOG_ROOT"
sbatch \
  --partition=minor \
  --gres=gpu:1 \
  --cpus-per-task=3 \
  --mem=16G \
  --time=2-00:00:00 \
  --output="$SLURM_LOG_ROOT/%x-%A_%a.out" \
  --error="$SLURM_LOG_ROOT/%x-%A_%a.err" \
  cluster/pvsg/unary_baselines.sbatch
```

Outputs are immutable directories below `$PVSG_RUN_ROOT/unary-seed0/`. Array tasks 0 through 4
correspond respectively to `linear-probe`, `fused-linear`, `p-direct`, `integral-none`, and
`integral-p-sa`.

Rerun seed 0 together with four additional seeds as one 25-task array. The rerun keeps every
condition on the same evaluation implementation, including the conditional identity/category
diagnostic:

```bash
sbatch \
  --partition=minor \
  --gres=gpu:1 \
  --cpus-per-task=3 \
  --mem=16G \
  --time=2-00:00:00 \
  --output="$SLURM_LOG_ROOT/%x-%A_%a.out" \
  --error="$SLURM_LOG_ROOT/%x-%A_%a.err" \
  cluster/pvsg/unary_multiseed.sbatch
```

Tasks 0--4 use seed 0, 5--9 seed 1, and so on through tasks 20--24 at seed 4; condition order
within each block matches the first seed-0 run. Outputs go to the separate immutable directories
`$PVSG_RUN_ROOT/unary-multiseed/seed0/` through `seed4/`. The array defaults to five concurrent
jobs.

## Run the corrected seed-0 pair Tensor Brain comparison

Submit the same two-task array once for each model family. Task 0 disables feedback and task 1
uses P-SA. The `original` family uses the paper's recurrent dynamic context and direct index
scores; the `qtb` family uses feed-forward evolution and softplus-bias scores. Both use the rich
source-plus-reviewed hierarchy vocabulary, joint identity/category/predicate supervision, and
the shared learned visual mapping `g`.

```bash
cd /nfs/data8/harjes/MASTER/tensor-brain
source cluster/pvsg/common.sh
mkdir -p "$SLURM_LOG_ROOT"
sbatch \
  --export=ALL,PAIR_TB_VARIANT=original \
  --partition=minor \
  --gres=gpu:1 \
  --cpus-per-task=3 \
  --mem=16G \
  --time=2-00:00:00 \
  --output="$SLURM_LOG_ROOT/%x-%A_%a.out" \
  --error="$SLURM_LOG_ROOT/%x-%A_%a.err" \
  cluster/pvsg/pair_baselines.sbatch

sbatch \
  --export=ALL,PAIR_TB_VARIANT=qtb \
  --partition=minor \
  --gres=gpu:1 \
  --cpus-per-task=3 \
  --mem=16G \
  --time=2-00:00:00 \
  --output="$SLURM_LOG_ROOT/%x-%A_%a.out" \
  --error="$SLURM_LOG_ROOT/%x-%A_%a.err" \
  cluster/pvsg/pair_baselines.sbatch
```

The models do not require the larger-memory partition at batch size 128. If `major` is used,
prefer a cluster-specific GPU constraint so every learned condition uses one GPU family. Each
`config.json` records the CUDA device and compute capability, and each Slurm log records the GPU
name and driver. Results are written below `$PVSG_RUN_ROOT/pair-corrected-original-seed0/` and
`$PVSG_RUN_ROOT/pair-corrected-qtb-seed0/`. The jobs use Adam at `1e-4`, a `1e-5` parameter-group
rate for `g`, and validation every 250 updates. These are two paper-motivated family bundles,
not a controlled one-factor evolution ablation: evolution and score parameterization both change.
