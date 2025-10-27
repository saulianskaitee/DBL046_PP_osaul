#!/bin/bash
#BSUB -q cabgpu
#BSUB -J cabgpu_check
#BSUB -u s232958@dtu.dk
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:05
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
#BSUB -oo runs/gpu_%J.out
#BSUB -eo runs/gpu_%J.err
#BSUB -gpu "num=1"

set -euo pipefail

echo "Node: $(hostname)"
echo "Job: $LSB_JOBID  Queue: $LSB_QUEUE"

# Ensure nvidia-smi exists and at least one GPU is visible
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "cab-gpu available"
  # Optional: brief GPU summary
  nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
  exit 0
else
  echo "cab-gpu not available"
  exit 1
fi
