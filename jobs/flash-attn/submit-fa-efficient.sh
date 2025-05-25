#!/bin/bash

#SBATCH --account=a-large-sc
#SBATCH --time=00:14:59
#SBATCH --job-name=flash-attn-efficient
#SBATCH --output=/iopsstor/scratch/cscs/%u/project/output/logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/iopsstor/scratch/cscs/ahkhan/project/envs/base_env.toml
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs
#SBATCH --partition debug

# Record start time
start_time=$(date)
echo "START TIME: $start_time --sl 4096"

# Set up ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
PROJECT_DIR="/iopsstor/scratch/cscs/$USER/project"

CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="python3 $PROJECT_DIR/train.py \
    --sequence-length 4096 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --lr-warmup-steps 100 \
    --training-steps 1000 \
    --fused-optimizer \
    --output-dir $PROJECT_DIR/output/data \
    --job-name $SLURM_JOB_NAME \
    --attention-backend efficient
    "

srun --cpus-per-task $SLURM_CPUS_PER_TASK bash -c "$CMD_PREFIX $TRAINING_CMD"

# Record end time
end_time=$(date)
echo "END TIME: $end_time"

# Calculate and print duration
duration=$(( $(date +%s) - $(date -d "$start_time" +%s) ))
minutes=$((duration / 60))
seconds=$((duration % 60))
echo "Duration: ${minutes} minutes and ${seconds} seconds"
