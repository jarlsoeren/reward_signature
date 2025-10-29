from bids import BIDSLayout
import nibabel as nb
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import scipy.io as sio
import matplotlib.pyplot as plt
import nibabel as nib

import os
from os.path import join, exists, split
import sys
import time
import urllib.request
import copy
import warnings
from tqdm import tqdm
from pprint import pprint
warnings.filterwarnings('ignore')

import glmsingle
from glmsingle.glmsingle import GLM_single

import nibabel as nib 
from nilearn.glm.first_level import compute_regressor 
from nilearn.glm.first_level import make_first_level_design_matrix 
import matplotlib.pyplot as plt 


from nilearn.image import mean_img, load_img
import os

db = BIDSLayout(root="/mnt/projects/rewardMap/STUDIES/pilotstudy/derivatives/", database_path='/mnt/projects/rewardMap/STUDIES/pilotstudy/derivatives/bids_layout/')

tmp_files = db.get(
        task='twostep',
        suffix="events",
        extension="tsv",
        return_type="filename"
    )

participants = [i.split('/')[-4].split('-')[-1] for i in tmp_files]

space_name = "MNI152NLin2009cAsym"
task = 'risksensitive'


def get_subject_data(pidx):
    """
    Loads bids compliant fMRI data for a single subject.

    Input:
        subject-id
    
    Returns:
        dictionary containing subject data

    """

    d = {}

    events_files = db.get(
        task=task,
        suffix="events",
        extension="tsv",
        return_type="filename",
        subject=participants[pidx]
    )

    d["event_files"] = events_files

    bold_files = db.get(
        task=task,
        suffix="bold",
        space=space_name,
        extension="nii.gz",
        return_type="filename",
        res="02",
        subject=participants[pidx]
    )

    # mask = db.get(
    #     task=task,
    #     suffix="mask",
    #     space=space_name,
    #     extension="nii.gz",
    #     return_type="filename",
    #     res="02",
    #     subject=participants[pidx]
    # )

    d["bold_files"] = bold_files

    tr=db.get(
        task=task,
        suffix="bold",
        space=space_name,
        extension="nii.gz",
        res="02",
        subject=participants[pidx]
    )[0].get_metadata()['RepetitionTime']

    d["tr"] = tr

    n_frames = nb.load(bold_files[0]).shape[-1]
    frame_times = np.arange(n_frames) * tr + tr /2

    d["n_frames"] = n_frames
    d["frame_times"] = frame_times

    return d


#subject_data = [get_subject_data(idx) for idx in range(len(participants))]
subject_data = [get_subject_data(idx) for idx in range(1)]
fmri_file = nib.load(subject_data[0]["bold_files"][0])

def make_des_mats(subject):
    frame_times = subject["frame_times"]
    #print(frame_times)
    
    events = pd.read_table(subject["event_files"][0])
    events.loc[pd.isna(events["duration"]), "duration"] = 0
    reward_onsets = events[events["event_type"] == "reward"]["onset"]

    dm = np.zeros((len(frame_times), 2))

    for i, onset in enumerate(reward_onsets):
        condition = i % 2
        idx = (np.abs(frame_times - onset)).argmin()
        dm[idx, condition] = 1

    return dm

for subject in subject_data:
    subject["design_matrices"] = make_des_mats(subject)

homedir = split(os.getcwd())[0]

datadir = join(homedir,'examples','data')
os.makedirs(datadir,exist_ok=True)

outputdir_glmsingle = join(homedir,'examples','example1outputs','GLMsingle')

opt = dict()

# set important fields for completeness (but these would be enabled by default)
opt['wantlibrary'] = 1
opt['wantglmdenoise'] = 1
opt['wantfracridge'] = 1

# for the purpose of this example we will keep the relevant outputs in memory
# and also save them to the disk
opt['wantfileoutputs'] = [1,1,1,1]
opt['wantmemoryoutputs'] = [1,1,1,1]

# running python GLMsingle involves creating a GLM_single object
# and then running the procedure using the .fit() routine
glmsingle_obj = GLM_single(opt)

# visualize all the hyperparameters
pprint(glmsingle_obj.params)


events = pd.read_table(subject_data[0]["event_files"][0])
events.loc[pd.isna(events["duration"]), "duration"] = 0
reward_events = events[events["event_type"] == "reward"] 

stimdur = reward_events["duration"].iloc[0]
tr = subject_data[0]["tr"]

print(f'There are 1 runs in total\n')
print(f'The stimulus duration is {stimdur} seconds\n')

# create a directory for saving GLMsingle outputs
outputdir_glmsingle = join(homedir,'examples','example1outputs','GLMsingle')

opt = dict()

# set important fields for completeness (but these would be enabled by default)
opt['wantlibrary'] = 1
opt['wantglmdenoise'] = 1
opt['wantfracridge'] = 1

# for the purpose of this example we will keep the relevant outputs in memory
# and also save them to the disk
opt['wantfileoutputs'] = [1,1,1,1]
opt['wantmemoryoutputs'] = [1,1,1,1]

# running python GLMsingle involves creating a GLM_single object
# and then running the procedure using the .fit() routine
glmsingle_obj = GLM_single(opt)

# visualize all the hyperparameters
pprint(glmsingle_obj.params)

start_time = time.time()

if not exists(outputdir_glmsingle):

    print(f'running GLMsingle...')
    
    # run GLMsingle
    results_glmsingle = glmsingle_obj.fit(
       subject_data[0]["design_matrices"],
       nib.load(subject_data[0]["bold_files"][0]).get_fdata(),
       stimdur,
       tr,
       outputdir=outputdir_glmsingle)
    
    # we assign outputs of GLMsingle to the "results_glmsingle" variable.
    # note that results_glmsingle['typea'] contains GLM estimates from an ONOFF model,
    # where all images are treated as the same condition. these estimates
    # could be potentially used to find cortical areas that respond to
    # visual stimuli. we want to compare beta weights between conditions
    # therefore we are not going to include the ONOFF betas in any analyses of 
    # voxel reliability
    
else:
    print(f'loading existing GLMsingle outputs from directory:\n\t{outputdir_glmsingle}')
    
    # load existing file outputs if they exist
    results_glmsingle = dict()
    results_glmsingle['typea'] = np.load(join(outputdir_glmsingle,'TYPEA_ONOFF.npy'),allow_pickle=True).item()
    results_glmsingle['typeb'] = np.load(join(outputdir_glmsingle,'TYPEB_FITHRF.npy'),allow_pickle=True).item()
    results_glmsingle['typec'] = np.load(join(outputdir_glmsingle,'TYPEC_FITHRF_GLMDENOISE.npy'),allow_pickle=True).item()
    results_glmsingle['typed'] = np.load(join(outputdir_glmsingle,'TYPED_FITHRF_GLMDENOISE_RR.npy'),allow_pickle=True).item()

elapsed_time = time.time() - start_time

print(
    '\telapsed time: ',
    f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
)
