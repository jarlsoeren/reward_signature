#!/bin/bash
#SBATCH --partition=HPC
#SBATCH --array=0-49
#SBATCH -o /mnt/scratch/projects/rewardMap/logs/boots_%A_%a.out
#SBATCH -J boots
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G

PYTHON_BIN=/mnt/projects/rewardMap/STUDIES/reward_signature/.venv/bin/python
COMPUTE_FOLDER='/mnt/projects/rewardMap/STUDIES/reward_signature/scripts'

# -----------------------------
# Define tasks
# -----------------------------
TASKS=(gonogo mid risksensitive hcp twostep)

# SLURM array index
ARRAY_ID=${SLURM_ARRAY_TASK_ID}

# 10 jobs per task
IDX=$((ARRAY_ID % 10))          # 0–9
TASK_ID=$((ARRAY_ID / 10))      # 0–4
TASK=${TASKS[$TASK_ID]}

echo "Running task=${TASK}, idx=${IDX}"

# -----------------------------
# Run script
# -----------------------------
$PYTHON_BIN \
  $COMPUTE_FOLDER/bootstrap_weights.py \
  --task ${TASK} \
  --idx ${IDX}

