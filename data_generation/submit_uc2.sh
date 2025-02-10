#!/bin/bash
#SBATCH --job-name=simulation_job    # Job name
#SBATCH --output=output_%j.log       # Output log file (%j = job ID)
#SBATCH --error=error_%j.log         # Error log file
#SBATCH --time=72:00:00              # Max runtime (hh:mm:ss)
#SBATCH --ntasks=1                     # One main task
#SBATCH --cpus-per-task=40            # Use 80 cores for multiprocessing
#SBATCH --mem=128G                     # Memory per node
#SBATCH --partition=single            # Partition (batch queue)
#SBATCH --mail-type=ALL              # Get email on job start, end, fail
#SBATCH --mail-user=pa6512@kit.edu  # Your email for notifications

export OMP_NUM_THREADS=$((${SLURM_CPUS_PER_TASK}/2))
# Load necessary modules (adjust as needed)
module load jupyter/ai

# Go to the directory where the script is located
cd $SLURM_SUBMIT_DIR

# Run your Python script
python data_generation_parallel.py

# Copy results from TMPDIR to home after job finishes
cp -r $TMPDIR/output_uc2 ~/PySSDR/data_generation