#!/bin/bash
### –- specify queue --
#BSUB -q cabgpu

### -- set the job Name --
#BSUB -J retrain_PPint

##BSUB -u your_email_address
#BSUB -u s232958@dtu.dk

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 4:00

# request 32GB of system-memory
#BSUB -R "rusage[mem=64GB]"
#BSUB -R "span[hosts=1]"

#BSUB -oo /zhome/c9/0/203261/DBL046_PP_osaul/DBL046_PP_osaul/tmp/logs/ppint_train_%J.out
#BSUB -ee /zhome/c9/0/203261/DBL046_PP_osaul/DBL046_PP_osaul/tmp/logs/ppint_train_%J.err

echo "Job starting on $(hostname)"
echo "Time is $(date)"
echo "LSB_JOBID is ${LSB_JOBID}"

########################
# 1. Activate environment
########################

# Activate your virtualenv / conda env:
source /work3/s232958/miniforge3/etc/profile.d/conda.sh
conda activate esm_gpu

########################
# 2. Set runtime env vars
########################

export WANDB_API_KEY=f8a6d759fe657b095d56bddbdb4d586dfaebd468
export WANDB_MODE=online
export WANDB_DIR=/zhome/c9/0/203261/DBL046_PP_osaul/DBL046_PP_osaul/tmp/wandb_cache
mkdir -p "$WANDB_DIR"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

########################
# 3. Go to working directory
########################

cd /zhome/c9/0/203261/DBL046_PP_osaul/DBL046_PP_osaul/tmp/ona_drafts

echo "Running in $(pwd)"
echo "Python is $(which python)"

########################
# 4. Run training
########################

python retrain_PPint.py

########################
# 5. Done
########################
echo "Job finished at $(date)"
