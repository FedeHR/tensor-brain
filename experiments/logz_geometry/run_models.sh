#!/bin/bash
# Sweep the log Z geometry probe over several trained readouts.
# Run from the worktree root.
PY=/Users/fede/.claude/jobs/4364cc95/tmp/probe-env/bin/python
mkdir -p output/logz
for M in "gpt2" "HuggingFaceTB/SmolLM2-135M" "Qwen/Qwen2.5-0.5B-Instruct"; do
  SLUG=$(echo "$M" | tr '/' '_')
  echo "=================== $M ==================="
  $PY experiments/logz_geometry/probe.py \
      --model "$M" --n-states 20000 --n-docs 300 --nulls --state-nulls \
      --out "output/logz/${SLUG}.json" 2>&1 | grep -vE "it/s\]|examples/s\]"
done
