#!/bin/sh
### General options
### –- specify queue --
#BSUB -q cabgpu
### -- set the job Name --
#BSUB -J pPL_for_PPint
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 01:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=15GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s232958@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

# ---------------------------------------------------------
# Load software modules
# ---------------------------------------------------------

module load cuda/11.6

# ---------------------------------------------------------
# Activate your conda / venv environment
# ---------------------------------------------------------

# If using conda:
source ~/.bashrc       # required for conda on DTU HPC
conda activate esm_cuda

# If using a Python venv:
# source /work3/s232958/envs/esm_inverse/bin/activate

# Print GPU info
echo "Running on:"
nvidia-smi

# ---------------------------------------------------------
# Run your script
# ---------------------------------------------------------

python3 /zhome/c9/0/203261/DBL046_PP_osaul/DBL046_PP_osaul/tmp/ona_drafts/avLL_pPL_PPint.py