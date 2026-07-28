#!/usr/bin/env bash
set -uo pipefail

if (( $# == 0 )); then
  echo "usage: $0 COMMAND [ARGUMENT ...]" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; cannot monitor GPU usage" >&2
  exit 2
fi

SAMPLE_FILE="${TMPDIR:-/tmp}/nvidia-smi.csv"
GPU_SELECTOR="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
NVIDIA_SMI_DEVICE=()
if [[ -n "$GPU_SELECTOR" ]]; then
  NVIDIA_SMI_DEVICE=(--id="$GPU_SELECTOR")
else
  echo "GPU monitor warning: no Slurm GPU selector found; sampling every visible GPU" >&2
fi

nvidia-smi "${NVIDIA_SMI_DEVICE[@]}" \
  --query-gpu=memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits \
  --loop-ms=1000 >"$SAMPLE_FILE" &
MONITOR_PID=$!

finish_monitor() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
  if [[ -s "$SAMPLE_FILE" ]]; then
    awk -F, '
      {
        for (field = 1; field <= NF; field++) {
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", $field)
        }
        if ($1 + 0 > peak_memory) peak_memory = $1 + 0
        if ($2 + 0 > total_memory) total_memory = $2 + 0
        if ($3 + 0 > peak_utilization) peak_utilization = $3 + 0
        utilization_sum += $3 + 0
        samples++
      }
      END {
        printf "GPU monitor: peak_used_mib=%d total_mib=%d average_utilization_percent=%.1f peak_utilization_percent=%d samples=%d\n", peak_memory, total_memory, utilization_sum / samples, peak_utilization, samples
      }
    ' "$SAMPLE_FILE"
    echo "GPU monitor samples: $SAMPLE_FILE"
  else
    echo "GPU monitor warning: nvidia-smi produced no samples" >&2
  fi
}
trap finish_monitor EXIT

"$@"
COMMAND_STATUS=$?
exit "$COMMAND_STATUS"
