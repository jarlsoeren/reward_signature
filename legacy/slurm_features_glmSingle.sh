#!/bin/bash
#SBATCH --partition=HPC
#SBATCH -o /mnt/scratch/projects/rewardMap/logs/glmSingle_%j.out
#SBATCH -J glmSingle
#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G


PYTHON_BIN=/mnt/scratch/projects/rewardMap/reward_signature/.venv/bin/python
ANACONDA_DIR=/mnt/projects/rewardMap/STUDIES/preregistration/.venv/

#DISPLAY=
#. $ANACONDA_DIR/etc/profile.d/conda.sh

COMPUTE_FOLDER='/mnt/scratch/projects/rewardMap/reward_signature/'

# activate the env your interested in

$PYTHON_BIN $COMPUTE_FOLDER/feature_extraction_glmSingle.py