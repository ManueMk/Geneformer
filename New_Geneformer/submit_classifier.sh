#!/bin/bash
#SBATCH --job-name=gene_full
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/pipeline_%j.log

# 1. Internet Access for Hugging Face
export https_proxy=http://squid.zib.de:3128
export http_proxy=http://squid.zib.de:3128

# 2. Define Paths
export ENV_DIR="/home/htc/mnakam/miniforge3/envs/geneformer_311"
export SITE_PACKAGES="$ENV_DIR/lib/python3.11/site-packages"
export PYTHON_EXEC="$ENV_DIR/bin/python"

# 3. Path Injection (This is what fixed your test command)
export PYTHONPATH="$SITE_PACKAGES"
export PYTHONNOUSERSITE=1

echo "Starting Job..."
echo "Using Python: $PYTHON_EXEC"
echo "Searching in: $PYTHONPATH"

# 4. Run the script 
# Make sure the filename matches your file (run_classifier.py or run_pipeline.py)
$PYTHON_EXEC run_classifier.py