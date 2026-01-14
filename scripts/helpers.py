import os
import random

import numpy as np
import nibabel as nb
import pandas as pd

from sklearn.preprocessing import StandardScaler
from nilearn.interfaces.fmriprep import load_confounds
from bids import BIDSLayout



def get_subs(dir):
    subs = []
    p = os.path.join(dir, "gonogo")
    i = 0

    for (root, dirs, files) in os.walk(p):
        for file in files:
            if file.startswith("betas"):
                subs.append(f"sub-{i:03d}")
                i += 1
            
    return subs

def get_events(pidx, db, task, participants):

    events_files = db.get(
        task=task,
        suffix="events",
        extension="tsv",
        return_type="filename",
        subject=participants[pidx]
    )



    return events_files

def load_data(tasks, masked_dir, map):
    db = BIDSLayout(root="/mnt/projects/rewardMap/STUDIES/pilotstudy/derivatives/", database_path='/mnt/projects/rewardMap/STUDIES/pilotstudy/derivatives/bids_layout/')

    dir = os.path.join(masked_dir, map)

    X_list = []
    Y_list = []

    subjects = get_subs(dir)

    index_ranges = {
        "tasks": {task: [] for task in tasks},
        "subjects": {sub: [] for sub in subjects},
        "task-subjects": {}
    }

    running_index = 0


    tmp_files = db.get(
            task=tasks[0],
            suffix="events",
            extension="tsv",
            return_type="filename"
        )

    participants = [i.split('/')[-4].split('-')[-1] for i in tmp_files]

    #trial_type = df[[df]'trial_type']
    #print(trial_type)
    trial_type_list = []


    for task in tasks:
        p = os.path.join(dir, task)
        for (root,dirs,files) in os.walk(p):
            for file in files:
                if file.startswith("betas"):
                    beta = np.load(os.path.join(root, file), mmap_mode='r')
                    X_list.append(beta)
                elif file.startswith("rewards"):
                    rewards = np.load(os.path.join(root, file), mmap_mode='r')
                    Y_list.append(rewards)
                    
                    n_trials = rewards.shape[0]
                    
                    idxs = list(range(running_index, running_index + n_trials))

                    index_ranges["tasks"][task] += idxs

                    sub = f"sub-{file.removesuffix('.npy').split('_')[2]}"
                    index_ranges["subjects"][sub] += idxs

                    df = pd.read_csv(get_events(int(sub.split('-')[-1]), db, task, participants)[0], sep="\t")
                    trial_type_list += df[df['event_type'] == 'reward']['trial_type'].to_list()

                    index_ranges["task-subjects"][f"{task}-{sub}"] = idxs

                    running_index += n_trials  

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)    

    return X, Y, trial_type_list, index_ranges

def training_splits(index_ranges, split="subs"):
    
    train_splits = []
    val_splits = []
    test_split = None

    if split == "subs":
        ranges = index_ranges["subjects"]
        #get test split
        # test_sub = random.choice(list(ranges.keys()))
        # test_split = ranges[test_sub]
        # ranges.pop(test_sub)

        #train + val split
        for sub in list(ranges.keys()):
            val_splits.append(ranges[sub])
            train_splits.append([])

            for sub2 in list(ranges.keys()):
                if sub == sub2:
                    continue
                train_splits[-1] += ranges[sub2]

    elif split == "tasks":
        ranges = index_ranges["tasks"]

        test_task = random.choice(list(ranges.keys()))
        test_split = ranges[test_task]
        ranges.pop(test_task)

        for task in list(ranges.keys()):
            val_splits.append(ranges[task])
            train_splits.append([])

            for task2 in list(ranges.keys()):
                if task == task2:
                    continue
                training_splits[-1] += ranges[task2]

    elif split == "sub_task":
        ranges = index_ranges["task-subjects"]       

    return train_splits, val_splits, test_split

def leave_one_sub_out_splits(index_ranges, task, tasks):
    train_splits = [] # train splits for one task: lists of indices of all subjects but one each list leaves one subject out
    test_splits = {} # val splits for one task: lists of indexes of the left out subject

    ranges = index_ranges["task-subjects"]


    for tsk in tasks:
        test_splits[tsk] = [ranges[task_sub] for task_sub in ranges if task_sub.startswith(tsk)]
    
    n_subs = len(test_splits[task])
    for i in range(n_subs):
        split = [idx
                for j in range(n_subs) if j != i
                for idx in test_splits[task][j]]        
        train_splits.append(split)
        
    return train_splits, test_splits


def individual_z_scoring(X, index_ranges):
    X = X.copy()
    ranges = index_ranges["task-subjects"]

    for key, idx in ranges.items():
        task_sub = X[idx]
        
        scaler = StandardScaler()
        X[idx] = scaler.fit_transform(task_sub)
    
    return X

load_data(["mid"], "brain_masked", "lss_maps")
