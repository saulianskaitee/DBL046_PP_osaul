#!/bin/bash
### –- specify queue --
#BSUB -q cabgpu

### -- set the job Name --
#BSUB -J av_LL_pPL_calc

##BSUB -u your_email_address
#BSUB -u s232958@dtu.dk

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00

# request 32GB of system-memory
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"

#BSUB -oo runs/%J.out
#BSUB -eo runs/%J.err

set -euo pipefail

echo "Node: $(hostname)"
echo "JobID: $LSB_JOBID"
echo "Queue: $LSB_QUEUE"
echo "CUDA_VISIBLE_DEVICES at submit time: ${CUDA_VISIBLE_DEVICES:-unset}"

# --- Load/activate environment ---
# (adjust this part to however you normally load PyTorch on the cluster)

# Example if you use conda:
source /work3/s232958/miniforge3/etc/profile.d/conda.sh
conda activate esm_gpu

# --- Go to your project directory ---
cd /zhome/c9/0/203261/DBL046_PP_osaul/DBL046_PP_osaul/tmp/

echo "Running in: $(pwd)"
echo "Python: $(which python)"
python --version
nvidia-smi || true

# --- Actually run the job ---
python avLL_pPL_PPint.py

echo "Job finished."
