import os

import pandas as pd
import nibabel as nb
import numpy as np

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
from nilearn.image import resample_to_img

from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from nilearn.datasets import fetch_atlas_pauli_2017

import numpy as np

def generalization_to_sub(data_dict):
    X_list = []
    Y_list = []

    X_test_list = []
    Y_test_list = []

    val_idxs = []
    subject_to_val_idx = {}  # A dictionary to map subjects to their validation indices

    tasks = list(data_dict.keys())
    test_sub = np.random.randint(0, 15)  # Random test subject index

    running_index = 0  # Keeps track of where in the big concatenated matrix we are

    for task in tasks:
        d = data_dict[task]

        for j, sub in enumerate(list(d.keys())):
            betas = d[sub]["betas"]       # shape: trials x voxels
            rewards = d[sub]["rewards"]   # shape: trials

            if j == test_sub:
                # Test subject -> goes into final test set (not used in CV)
                X_test_list.append(betas)
                Y_test_list.append(rewards)
            else:
                n_trials = len(rewards)
                # If the subject hasn't been added to the validation list yet, initialize the entry
                if sub not in subject_to_val_idx:
                    subject_to_val_idx[sub] = []

                # Append the indices for this subject (across all tasks) to the validation list
                subject_to_val_idx[sub].append(np.arange(running_index, running_index + n_trials))

                # Add to full train set
                X_list.append(betas)
                Y_list.append(rewards)

                running_index += n_trials

    # Now concatenate everything
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)

    X_test = np.concatenate(X_test_list, axis=0)
    Y_test = np.concatenate(Y_test_list, axis=0)

    # --- Build training index sets for each validation split ---
    all_idx = np.arange(len(X))

    # Create a list of training indices, excluding indices for the validation subjects
    train_idxs = []
    for sub, val_idx_list in subject_to_val_idx.items():
        # Flatten the validation indices for this subject across tasks
        val_idx = np.concatenate(val_idx_list)
        train_idx = np.setdiff1d(all_idx, val_idx)
        train_idxs.append(train_idx)
        val_idxs.append(val_idx)

    return X, Y, X_test, Y_test, train_idxs, val_idxs



def run_model(train_tasks, test_tasks, n_subs, maps_dir):

    atlas = fetch_atlas_pauli_2017(atlas_type="deterministic")

    img = nb.load(atlas.maps)
    data = img.get_fdata()

    putamen_mask = np.isin(data, 1).astype(np.int8)
    caudate_mask = np.isin(data, 2).astype(np.int8)
    nucleus_accumbens_mask = np.isin(data, 3).astype(np.int8)

    combined_mask = putamen_mask + caudate_mask + nucleus_accumbens_mask
    combined_mask_img = nb.Nifti1Image(combined_mask, img.affine, img.header)

    tasks = train_tasks + test_tasks

    data_dict = {task : {} for task in tasks}

    ## load and mask data
    for task in tasks:
        p = os.path.join(maps_dir, task)
        subject_dirs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]

        for i, sub in enumerate(subject_dirs):
            if i >= n_subs:
                break

            sub_dir_p = os.path.join(p, sub)

            img_files = []
            rewards_file = None

            for roor, dirs, files in os.walk(sub_dir_p):
                for file in files:
                    if file.endswith(".nii.gz"):
                        img_files.append(file)
                    elif file.startswith("rewards"):
                        rewards_file = os.path.join(sub_dir_p, file)

            imgs = [nb.load(os.path.join(sub_dir_p, f)) for f in sorted(img_files)]
            imgs = concat_imgs(imgs)
            combined_mask_img = resample_to_img(combined_mask_img, imgs, interpolation='nearest', force_resample=True, copy_header=True)

            masked_data = apply_mask(imgs=imgs, mask_img=combined_mask_img)

            if rewards_file != None: #just bc the lss-map creation is still running
                rewards = pd.read_csv(rewards_file)
                rewards = rewards["reward"].to_numpy()

            data_dict[task][sub] = {"betas": masked_data, "rewards": rewards}
        
    X, Y, X_test, Y_test, train_idxs, val_idxs = generalization_to_sub(data_dict)

    Y = np.sign(Y)
    Y_test = np.sign(Y_test)



    print("Data shapes")
    print(f"Train: {X.shape}\nTest: {X_test.shape}\n\n")
    classes, count = np.unique_counts(Y)
    print(f"Baseline accuracies of train: {classes} - {np.round(count/np.sum(count), 2)}")
    classes, count = np.unique_counts(Y_test)
    print(f"Baseline accuracies of test: {classes} - {np.round(count/np.sum(count), 2)}\n")
    print(f"Validation splits: {len(val_idxs)}")

    params = {
        "max_depth": [5, 10, 15, 20, 50],
        "n_estimators": [100, 200]
    }

    best_model = None
    best_acc = -1

    n_Folds = 4

    for depth in params["max_depth"]:
        for n in params["n_estimators"]:
            print(f"Running random forest for max_depth: {depth} and n_estimators: {n}")

            fold_train_accuracy = []
            fold_val_accuracy = []
            for i in range(n_Folds):
                print(f"Fold{i}/{n_Folds}")

                print(f"val: {val_idxs[i]}")
                print(f"train: {train_idxs[i]}")

                X_train = X[train_idxs[i]]
                Y_train = Y[train_idxs[i]]

                X_val = X[val_idxs[i]]
                Y_val = Y[val_idxs[i]]
                
                model = RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=42)

                model.fit(X_train, Y_train)
                y_pred = model.predict(X_train)
                fold_train_accuracy.append(balanced_accuracy_score(Y_train, y_pred))
                y_pred = model.predict(X_val)
                fold_val_accuracy.append(balanced_accuracy_score(Y_val, y_pred))

            print(f"Train accuracies: {fold_train_accuracy}, Average: {np.average(fold_train_accuracy)}")
            val_acc = np.average(fold_val_accuracy)
            print(f"Val accuracies: {fold_val_accuracy}, Average: {val_acc}")
            print("\n")

            if val_acc > best_acc:
                best_model = model


    print(f"Best model: {best_model.get_params()}")
    y_pred = best_model.predict(X_test)
    print(f"Test accuracy: {balanced_accuracy_score(Y_test, y_pred)}")
    


if __name__ == "__main__":
    tasks = ['gonogo', 'hcp', 'mid', 'risksensitive', 'twostep'] #posner
    n_subs = 15
    maps_dir = './lss_maps'

    train_tasks = tasks[:3]
    test_tasks = tasks[3:]

    run_model(train_tasks, test_tasks, n_subs, maps_dir)
    

    

