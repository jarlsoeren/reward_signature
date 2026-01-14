#!/bin/bash
#SBATCH --partition=HPC
#SBATCH --array=0-75
#SBATCH --output=/mnt/scratch/projects/rewardMap/logs/final_%A_%a.out
#SBATCH --job-name=final_nested
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=7-0

PYTHON_BIN=/mnt/projects/rewardMap/STUDIES/reward_signature/.venv/bin/python
COMPUTE_FOLDER='/mnt/projects/rewardMap/STUDIES/reward_signature/scripts'
PARAM_DIR='/mnt/projects/rewardMap/STUDIES/reward_signature/job_params'

# Job JSON file for this array index
PARAM_FILE="$PARAM_DIR/job_${SLURM_ARRAY_TASK_ID}.json"

echo "Running job $SLURM_ARRAY_TASK_ID"
echo "Loading parameters from $PARAM_FILE"

$PYTHON_BIN $COMPUTE_FOLDER/nested_cross_final.py --json_params "$PARAM_FILE"
