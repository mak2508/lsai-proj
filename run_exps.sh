#!/bin/bash

# List of job scripts to submit
JOBS=(
    "jobs/baseline/submit-baseline.sh"
    "jobs/flash-attn/submit-fa-cudnn.sh"
    "jobs/flash-attn/submit-fa-efficient.sh"
    "jobs/flash-attn/submit-fa-fa3.sh"
    "jobs/flash-attn/submit-fa.sh"
    "jobs/fp8/submit-fp8-row.sh"
    "jobs/fp8/submit-fp8.sh"
    # Add more job scripts as needed
)

# Function to check number of jobs in system
check_jobs() {
    # Count running jobs
    running=$(squeue -u $USER -t R | wc -l)
    # Count pending jobs
    pending=$(squeue -u $USER -t PD | wc -l)
    # Subtract 1 from each to account for header line
    running=$((running - 1))
    pending=$((pending - 1))
    # Return total jobs in system
    echo $((running + pending))
}

# Submit jobs one by one
for job in "${JOBS[@]}"; do
    # Wait until we have less than 2 jobs in the system
    while [ $(check_jobs) -ge 2 ]; do
        echo "Waiting for job slots to become available..."
        sleep 30  # Check every 30 seconds
    done
    
    echo "Submitting $job..."
    sbatch $job
    
    # Small delay to ensure SLURM has time to process the submission
    sleep 5
done

echo "All jobs submitted!"