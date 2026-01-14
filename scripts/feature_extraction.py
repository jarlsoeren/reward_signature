import os
import pandas as pd
import numpy as np
import nibabel as nb



from bids import BIDSLayout
from nilearn.interfaces.fmriprep import load_confounds
from tqdm import tqdm
from nilearn.glm.first_level import FirstLevelModel


def get_subject_data(pidx, db, task, participants, space_name):
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

    d["confounds"] = load_confounds(bold_files, strategy=("motion", "wm_csf"))[0]

    return d



def lss_transformer(events, row_number):
    """
        Transforms events dataframe from behavioral data into a events matrix for LSS maps.
        With one column containing the onset for specified trial and one column containing all others.

        Input:
            events: Dataframe
            row_number: integer

        Returns:
            reward_events: dataframe with specified columns
            trial_name: string
            y: vector containig rewards given    
    """
    events = events.copy()

    #events with nan duration are of len 0
    events.loc[pd.isna(events["duration"]), "duration"] = 0

    reward_events = events[events["event_type"] == "reward"]
    reward_events = reward_events[["onset", "duration"]].reset_index(drop=True)
    reward_events["trial_type"] = "reward"

    trial_name = f"reward_{row_number:03d}"

    reward_events.loc[row_number, "trial_type"] = trial_name


    return reward_events, trial_name   


    

def extract_betas_lss(subject, id, output_dir, out_type):

    glm_parameters = {
        "t_r": subject["tr"],
        "hrf_model": "spm",
        "drift_model": "cosine",
        "noise_model": "ar1",
        "high_pass": 0.008, 
        "smoothing_fwhm": 6,
        "n_jobs": -1,
        "slice_time_ref": 0.5,
    }


    events = pd.read_table(subject["event_files"][0])
    n_trials = len(events[events["event_type"] == "reward"])
    fmri_file = nb.load(subject["bold_files"][0]) #for some reason this makes shit crahs lol

    for i_trial in tqdm(range(n_trials)):
        lss_events_df, trial_condition = lss_transformer(events, i_trial)

        # Compute and collect beta maps
        lss_glm = FirstLevelModel(**glm_parameters)
        lss_glm.fit(fmri_file, lss_events_df, confounds=subject["confounds"])
    
        beta_map = lss_glm.compute_contrast(
            trial_condition,
            output_type=out_type,
        )

        save_maps(beta_map, lss_glm.design_matrices_[0], id, i_trial, output_dir)

    # save rewards
    sub_dir = os.path.join(output_dir, f"sub-{id:03d}")

    rewards_file = os.path.join(sub_dir, f"rewards_sub-{id:03d}.csv")

    rewards = events[events["event_type"] == "reward"][["trial", "reward"]].reset_index(drop=True)
    # Convert to a DataFrame for clarity and easy loading later
    rewards.to_csv(rewards_file)
    
def save_maps(map, dm, subject_id, trial, output_dir):
    """
        saves the lss maps and design matrices.

        Inputs:
            map: beta image
            dm: design matrix
            subject_id: integer
            trial: integer

    """
    sub_dir = os.path.join(output_dir, f"sub-{subject_id:03d}")
    beta_file = os.path.join(sub_dir, f"beta_map_trial-{trial:03d}.nii.gz")
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)

    nb.save(map, beta_file)

    dm_file = os.path.join(sub_dir, f"des_mat_trial-{trial:03d}.csv")
    dm.to_csv(dm_file, index=False)
    

def extract_features(task, out_type, out_dir):
    db = BIDSLayout(root="/mnt/projects/rewardMap/STUDIES/pilotstudy/derivatives/", database_path='/mnt/projects/rewardMap/STUDIES/pilotstudy/derivatives/bids_layout/')

    tmp_files = db.get(
            task=task,
            suffix="events",
            extension="tsv",
            return_type="filename"
        )

    participants = [i.split('/')[-4].split('-')[-1] for i in tmp_files]
    space_name = "MNI152NLin2009cAsym"


    output_dir = os.path.join(os.path.curdir, out_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_dir = os.path.join(output_dir, task)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for idx in range(12, len(participants)):
        subject = get_subject_data(idx, db, task, participants, space_name)
        extract_betas_lss(subject, idx, output_dir, out_type)
