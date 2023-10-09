#!/bin/bash
#SBATCH --nodes=1 # Get one node
#SBATCH --cpus-per-task=4 # Two cores per task
#SBATCH --ntasks=1 # But only one task
#SBATCH --gres=gpu:2 # And two GPUs
#SBATCH --gres-flags=enforce-binding # Insist on good CPU/GPU alignment
#SBATCH --time=23:59:59 # Run for 1 day, at most
#SBATCH --job-name=PACL  # Name the job so I can see it in squeue
#SBATCH --mail-type=BEGIN,END,FAIL # Send me email for various states
#SBATCH --mail-user ma649596@ucf.edu # Use this address

# Load modules
module load anaconda/anaconda3

source /apps/anaconda/anaconda3/etc/profile.d/conda.sh

conda activate pacl

export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "include-system-site-packages=false" >> $CONDA_PREFIX/pyvenv.cfg
export PYTHONPATH=${PYTHONPATH}:${HOME}/.local/lib/python3.10/site-packages


cd src
python -m training.main \
    --imagenet-val /datasets/ImageNet2012nonpub/validation/ \
    --model ViT-B-32-quickgelu \
    --pretrained laion400m_e32 \
    --args pacl
