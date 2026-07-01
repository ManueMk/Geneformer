#!/bin/bash
#SBATCH --job-name=gene_isp
#SBATCH --time=24:00:00            # ISP takes a long time
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8          # Increased CPUs for nproc=4
#SBATCH --mem=64G
#SBATCH --output=logs/isp_%j.log

export ENV_DIR="/home/htc/mnakam/miniforge3/envs/geneformer_311"
export PYTHON_EXEC="$ENV_DIR/bin/python"
export PYTHONNOUSERSITE=1

# Run the perturbation script
$PYTHON_EXEC run_insilico.py