#!/bin/bash
#SBATCH --partition=HPC
#SBATCH -o /mnt/scratch/projects/rewardMap/logs/lss_%j.out
#SBATCH -J lss
#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G


PYTHON_BIN=/mnt/projects/rewardMap/STUDIES/reward_signature/.venv/bin/python
ANACONDA_DIR=/mnt/projects/rewardMap/STUDIES/preregistration/.venv/

#DISPLAY=
#. $ANACONDA_DIR/etc/profile.d/conda.sh

COMPUTE_FOLDER='/mnt/projects/rewardMap/STUDIES/reward_signature/scripts'

# activate the env your interested in

$PYTHON_BIN $COMPUTE_FOLDER/call_feature_extraction.py