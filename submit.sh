#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=output.log
#SBATCH --ntasks=1
#SBATCH --time=02:00:00  # Set the maximum runtime here

# Load the Conda module if needed (depends on your cluster setup)
module load jupyter/ai

# Activate the SSDR conda environment
source activate SSDR

# Run the Python script
python simulation_generate_data.py