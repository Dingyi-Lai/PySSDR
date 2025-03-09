#!/bin/bash
#SBATCH --job-name=simulation_job    # Job name
#SBATCH --output=output_%j.log       # Output log file (%j = job ID)
#SBATCH --error=error_%j.log         # Error log file
#SBATCH --time=72:00:00              # Max runtime (hh:mm:ss)
#SBATCH --ntasks=1                     # One main task
#SBATCH --cpus-per-task=120            # Use 152 cores for multiprocessing
#SBATCH --mem=80G                     # Memory per node
#SBATCH --partition=normal            # Partition (batch queue)
#SBATCH --mail-type=ALL              # Get email on job start, end, fail
#SBATCH --mail-user=pa6512@kit.edu  # Your email for notifications

export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS=1            # Set to 1 to avoid thread oversubscription in nested parallelism

# Load necessary modules (adjust as needed)
module load jupyter/ai

# Go to the directory where the script is located
cd $SLURM_SUBMIT_DIR

# Run your Python script
python data_generation_parallel_generalized.py

# Copy results from TMPDIR to home after job finishes
cp -r $TMPDIR/output_modified_wo_unstructured2 ~/PySSDR/data_generation