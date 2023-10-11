#!/bin/bash
#SBATCH --nodes=1 # Get one node
#SBATCH --cpus-per-task=4 # Two cores per task
#SBATCH --ntasks=1 # But only one task
#SBATCH --gres=gpu:2 # And two GPUs
#SBATCH --gres-flags=enforce-binding # Insist on good CPU/GPU alignment
#SBATCH --time=72:00:00 # Run for 1 day, at most
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

python -m pip install tensorboard

cd src
    
python -m training.main \
    --save-frequency 1 \
    --report-to tensorboard \
    --train-data="../../cc3/cc3m/{00000..00331}.tar" \
    --warmup 2000 \
    --batch-size=256 \
    --epochs=10 \
    --workers=2 \
    --model ViT-B-16 \
    --name "ViT-B-16-PACL" \
    --seed 0 \
    --local-loss \
    --gather-with-grad \
    --imagenet-val /datasets/ImageNet2012nonpub/validation/ \
    --train-num-samples 10000 \
    --pacl \
    --pretrained laion400m_e32 \
    --lr 0.0005 \
    --eps 0.000001 \
    --beta1 0.9 \
    --beta2 0.98 \
    --wd 0.2 \
    --dataset-resampled

