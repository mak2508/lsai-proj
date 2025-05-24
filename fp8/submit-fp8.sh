#!/bin/bash

#SBATCH --account=a-large-sc
#SBATCH --time=00:14:59
#SBATCH --job-name=fp8-te
#SBATCH --output=/iopsstor/scratch/cscs/%u/project/logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/iopsstor/scratch/cscs/ahkhan/project/envs/base_env.toml
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs
#SBATCH --partition debug

echo "START TIME: $(date) --sl 4096"

# Set up ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
PROJECT_DIR="/iopsstor/scratch/cscs/$USER/project/fp8"

CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="accelerate launch \
    --mixed_precision=fp8 \
    --num_processes 1 \
    $PROJECT_DIR/train.py \
    --fp8-train TE \
    --sequence-length 4096 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --lr-warmup-steps 100 \
    --training-steps 1000 \
    --fused-optimizer \
    "

srun --cpus-per-task $SLURM_CPUS_PER_TASK bash -c "$CMD_PREFIX $TRAINING_CMD"

echo "END TIME: $(date)"