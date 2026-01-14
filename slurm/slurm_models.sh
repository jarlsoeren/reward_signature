#!/bin/bash
#SBATCH --partition=HPC
#SBATCH -o /mnt/scratch/projects/rewardMap/logs/final_gonogo%j.out
#SBATCH -J final_gonogo
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G


PYTHON_BIN=/mnt/projects/rewardMap/STUDIES/reward_signature/.venv/bin/python
ANACONDA_DIR=/mnt/projects/rewardMap/STUDIES/preregistration/.venv/

#DISPLAY=
#. $ANACONDA_DIR/etc/profile.d/conda.sh

COMPUTE_FOLDER='/mnt/projects/rewardMap/STUDIES/reward_signature/scripts'

# activate the env your interested in

$PYTHON_BIN $COMPUTE_FOLDER/final_model.py