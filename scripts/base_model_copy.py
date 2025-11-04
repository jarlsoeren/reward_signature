import os

import pandas as pd
import nibabel as nb
import numpy as np
import random

from nilearn.image import concat_imgs
from nilearn.masking import apply_mask


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

masked_dir = "/mnt/projects/rewardMap/STUDIES/reward_signature/brain_masked"

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

def load_data(tasks, map):
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

                    index_ranges["task-subjects"][f"{task}-{sub}"] = idxs

                    running_index += n_trials  

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)    

    return X, Y, index_ranges

def training_splits(index_ranges, split="subs"):
    
    train_splits = []
    val_splits = []
    test_split = None

    if split == "subs":
        ranges = index_ranges["subjects"]
        #get test split
        test_sub = random.choice(list(ranges.keys()))
        test_split = ranges[test_sub]
        ranges.pop(test_sub)

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
        pass

    return train_splits, val_splits, test_split

def individual_z_scoring(X, index_ranges):
    X = X.copy()
    ranges = index_ranges["task-subjects"]

    for key, idx in ranges.items():
        task_sub = X[idx]
        
        scaler = StandardScaler()
        X[idx] = scaler.fit_transform(task_sub)
    
    return X



def run_model(X, Y, train_split, val_split, test_split):
    params = {
        "pca__n_components": [10, 25, 50],
        "pca__whiten": [True, False],
        "model__C": [0.1, 1, 10],
        "model__max_iter": [1000, 5000, 10000],
        "model__penalty": [None, "l1", "l2"]
    }

    X_test = X[test_split]
    y_test = Y[test_split]


    pipe = Pipeline([
    #    ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("model", LogisticRegression(solver='saga', random_state=42))
    ])

    cv_splits = list(zip(train_split, val_split))[:5]

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=params,
        n_iter=20,
        scoring='balanced_accuracy',
        cv=cv_splits,
        verbose=2,
        n_jobs=1,
        random_state=42,
        return_train_score=True
    )

    # Fit model on full data (search will use your custom CV indices)
    search.fit(X, Y)

     # Evaluate on held-out test set
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = balanced_accuracy_score(y_test, y_pred)

    results = search.cv_results_
    for mean_train, mean_val, params in zip(results['mean_train_score'],
                                            results['mean_test_score'],
                                            results['params']):
        print(f"{params} -> Train Acc: {mean_train:.4f}, Val Acc: {mean_val:.4f}")

    print("Best params:", search.best_params_)
    print("Validation accuracy:", search.best_score_)
    print("Test accuracy:", acc)




if __name__ == "__main__":
    tasks = ['gonogo', 'hcp', 'mid', 'risksensitive', 'twostep']
    maps = ['lss_maps', 'lss_t_maps', 'lss_z_maps']
    splits = ["subs", "tasks", "sub_task"]

    map = maps[0]

    X, Y, index_ranges = load_data(tasks, map)

    X = individual_z_scoring(X, index_ranges)
    Y = np.sign(Y)

    train, val, test = training_splits(index_ranges, split=splits[0])

    run_model(X, Y, train, val, test)
    # print(Y.shape)
    # print(f"Train: {len(train)} {len(train[0])}\n\n")
    # print(f"Val: {len(val)} {len(val[0])}\n\n")
    # print(f"Test: {len(test)}\n\n")

