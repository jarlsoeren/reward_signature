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
            combined_mask_img = resample_to_img(combined_mask_img, imgs, interpolation='nearest', force_resample=True)

            masked_data = apply_mask(imgs=imgs, mask_img=combined_mask_img)

            if rewards_file != None: #just bc the lss-map creation is still running
                rewards = pd.read_csv(rewards_file)
                rewards = rewards["reward"].to_numpy()

            data_dict[task][sub] = {"betas": masked_data, "rewards": rewards}
        
# prepare data for sklearn
    # shape = trials (time) x voxels
    X_list = []
    Y_list = []

    X_test_list = []
    Y_test_list = []

    tasks = list(data_dict.keys())
    test_task_idx = len(train_tasks) - 1  # leave last task out

    for i, task in enumerate(tasks):
        if task in train_tasks:
            d = data_dict[task]

            for sub in list(d.keys())[:1]:
                betas = d[sub]["betas"]
                rewards = d[sub]["rewards"]

                X_list.append(betas)
                Y_list.append(rewards)
        elif task in test_tasks:
            d = data_dict[task]
            
            for sub in list(d.keys())[:1]:
                betas = d[sub]["betas"]
                rewards = d[sub]["rewards"]

                X_test_list.append(betas)
                Y_test_list.append(rewards)


    # concatenate along trials axis only once
    X = np.concatenate(X_list, axis=0)
    Y = np.sign(np.concatenate(Y_list, axis=0))

    X_test = np.concatenate(X_test_list, axis=0)
    Y_test = np.sign(np.concatenate(Y_test_list, axis=0))

    print(X.shape)

    #model
    classes, count = np.unique_counts(Y)
    print(f"Baseline accuracies of train: {classes} - {np.round(count/np.sum(count), 2)}")
    classes, count = np.unique_counts(Y_test)
    print(f"Baseline accuracies of test: {classes} - {np.round(count/np.sum(count), 2)}")
    
    params = {
        "model__n_estimators": [100, 300, 500],
        "model__max_depth": [10, 30, 50, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.2],
        "model__bootstrap": [True, False]
    }

    pipe = Pipeline([
        #("scaler", StandardScaler()),
        #("pca", PCA(n_components=25)),
        #("model", LogisticRegression(solver='saga', penalty="l1", max_iter=10000, random_state=0))
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    print(pipe.get_params())

    search = RandomizedSearchCV(
        pipe,
        param_distributions=params,
        n_iter=15,  # try 30 random combinations
        scoring="balanced_accuracy",  # or 'f1_macro' etc.
        cv=5,
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )

    search.fit(X, Y)

    # Extract and display training + validation scores for all runs
    results = search.cv_results_
    for mean_train, mean_val, params in zip(results['mean_train_score'],
                                            results['mean_test_score'],
                                            results['params']):
        print(f"{params} -> Train Acc: {mean_train:.4f}, Val Acc: {mean_val:.4f}")

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_score = balanced_accuracy_score(Y_test, y_pred)
    print("\nTest Accuracy:", test_score)


    # # K-Fold cross-validation
    # splits = 5
    # kf = KFold(n_splits=splits, shuffle=True, random_state=42)

    # train_scores = []
    # val_scores = []

    # for i, (train_index, test_index) in enumerate(kf.split(X)):
    #     X_tr = X[train_index]
    #     X_val = X[test_index]
    #     y_tr = Y[train_index]
    #     y_val = Y[test_index]

    #     pipe.fit(X_tr, y_tr)


    #     y_pred = pipe.predict(X_tr)
    #     train_score = balanced_accuracy_score(y_tr, y_pred)
    #     train_scores.append(train_score)

    #     y_pred = pipe.predict(X_val)
    #     val_score = balanced_accuracy_score(y_val, y_pred)
    #     val_scores.append(val_score)
        
    #     print(f"Fold {i+1}/{splits} - Training Accuracy: {train_score:.4f}")
    #     print(f"Fold {i+1}/{splits} - Validation Accuracy: {val_score:.4f}")

    # print(f"\nMean training accuracy: {np.mean(train_scores):.4f}")
    # print(f"\nMean validation accuracy: {np.mean(val_scores):.4f}")

    # y_pred = pipe.predict(X_test)
    # test_score = balanced_accuracy_score(Y_test, y_pred)
        
    # print(f"Test Accuracy: {test_score:.4f}")



if __name__ == "__main__":
    tasks = ['gonogo', 'hcp', 'mid', 'risksensitive', 'twostep'] #posner
    n_subs = 15
    maps_dir = './lss_maps'

    train_tasks = tasks[:3]
    test_tasks = tasks[3:]

    run_model(train_tasks, test_tasks, n_subs, maps_dir)
    

    

