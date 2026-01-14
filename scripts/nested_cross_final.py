import numpy as np
import pandas as pd
import json
import argparse
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import balanced_accuracy_score, f1_score, make_scorer
from correlateTransformer import CorrelateTransformer

from helpers import load_data, leave_one_sub_out_splits

# PARAMETERS
TASKS = ['risksensitive', 'twostep', 'hcp', 'mid', 'gonogo']
MASK = "cut_brain_masked"
MAP = "lss_z_maps"
N_SPLITS = 4


CLASSIFICATION_SCORING = {   
    "balanced_acc": make_scorer(balanced_accuracy_score),
    "macro_f1": make_scorer(f1_score, average="macro")
}

def remove_samples(task, tsk, X, Y):
    # go-no-go, hcp, mid = 3 classes
    # risksensitive, twostep = 2 classes
    
    if task == 'risksensitive' or task == 'twostep':
        if tsk == 'gonogo' or tsk == 'hcp' or tsk == 'mid':
            mask = Y != -1
            X_new = X[mask]
            Y_new = Y[mask]

            return X_new, Y_new
        
    return X, Y
                

def run(X, Y, train_splits, test_splits, sub, task, other_tasks, params):

    pipe = Pipeline([
        ("scaler", StandardScaler()),   # gets replaced by passthrough or another scaler
        ("correlate", CorrelateTransformer()),
        ("pca", PCA()),                 # gets replaced by passthrough
        ("model", LogisticRegression(solver="saga", max_iter=10000, class_weight="balanced"))  # parameters get replaced
    ])

    
    refit_metric = list(CLASSIFICATION_SCORING.keys())[1]

    test_scores = {'train_task': [], 'test_task':[], 'test_subject': [], 'class_dist': [], 'balanced_acc': [], 'macro_f1': []}
    inner_cv_results = None


    X_train = X[train_splits[sub]]
    Y_train = Y[train_splits[sub]]

    X_test = X[test_splits[task][sub]]
    Y_test = Y[test_splits[task][sub]]

    inner_cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    print("RUNNING CV")

    # inner cross-validation for hyperparameter tuning
    search = GridSearchCV(
        estimator=pipe,
        param_grid=params,
        scoring=CLASSIFICATION_SCORING,
        refit=refit_metric,
        cv=inner_cv,
        verbose=4,
        n_jobs=-1,    
        return_train_score=True     
    )

    search.fit(X_train, Y_train)

    inner_cv_results = pd.DataFrame(search.cv_results_)
    best_model = search.best_estimator_

    # test on the left-out subject
    y_pred = best_model.predict(X_test) 
    test_balanced_acc = balanced_accuracy_score(Y_test, y_pred)
    test_macro_f1 = f1_score(Y_test, y_pred, average="macro")

    classes, counts = np.unique(Y_test, return_counts=True)
    counts = np.round(counts/np.sum(counts), 2)

    test_scores['train_task'].append(task)
    test_scores['test_task'].append(task)
    test_scores['test_subject'].append(sub)
    test_scores['class_dist'].append((classes, counts))
    test_scores['balanced_acc'].append(test_balanced_acc)
    test_scores['macro_f1'].append(test_macro_f1)

    # test on other tasks' left-out subject
    for tsk in other_tasks:
        X_other = X[test_splits[tsk][sub]]
        Y_other = Y[test_splits[tsk][sub]]

        X_other, Y_other = remove_samples(task, tsk, X_other, Y_other)

        y_other_pred = best_model.predict(X_other)
        other_balanced_acc = balanced_accuracy_score(Y_other, y_other_pred)
        other_macro_f1 = f1_score(Y_other, y_other_pred, average="macro")

        classes, counts = np.unique(Y_other, return_counts=True)
        counts = np.round(counts/np.sum(counts), 2)

        test_scores['train_task'].append(task)
        test_scores['test_task'].append(tsk)
        test_scores['test_subject'].append(sub)
        test_scores['class_dist'].append((classes, counts))
        test_scores['balanced_acc'].append(other_balanced_acc)
        test_scores['macro_f1'].append(other_macro_f1)

    test_scores = pd.DataFrame(test_scores)

    return test_scores, inner_cv_results, best_model

def resolve_objects(param_grid):
    resolved = []

    for cfg in param_grid:
        new_cfg = {}

        for key, value in cfg.items():
            new_values = []
            if key == "scaler":
                if value == "standard":
                    new_values.append(StandardScaler())
                elif value == "passthrough":
                    new_values.append("passthrough")

            elif key == "pca":
                if value == "pca":
                    new_values.append(PCA())
                elif value == "passthrough":
                    new_values.append("passthrough")

            else:
                new_values.append(value)

            new_cfg[key] = new_values

        resolved.append(new_cfg)

    return resolved
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_params", type=str, required=True)
    args = parser.parse_args()

    with open(args.json_params, "r") as f:
        job_data = json.load(f)

    task = job_data["task"]
    sub = job_data["sub"]
    params = job_data["params"]

    params = resolve_objects(params)
    
    print(f"\nLOADING DATA\n")
    X, Y, trial_types, index_ranges = load_data(TASKS, MASK, MAP)
    train_splits, test_splits = leave_one_sub_out_splits(index_ranges, task, TASKS)

    Y_signed = np.sign(Y)

    other_tasks = [t for t in TASKS if t != task]

    test_scores, inner_cv_results, best_model = run(X, Y_signed, train_splits, test_splits, sub, task, other_tasks, params)


    # Save results
    save_dir = f"/mnt/scratch/projects/rewardMap/reward_signature/results/{task}/"
    os.makedirs(save_dir, exist_ok=True)
    # specific test sets can identifies by subject and task like X[test_splits[task][sub]]
    # only data lost is the train/val splits of the inner CV, which can be regenerated
    np.savez_compressed(
        os.path.join(save_dir, f"data_sub{sub}.npz"),
        X=X,
        Y=Y,
        Y_signed=Y_signed,
        trial_types=np.array(trial_types),
        index_ranges=np.array(index_ranges, dtype=object),
        train_splits=np.array(train_splits, dtype=object),
        test_splits=np.array(test_splits, dtype=object),
    )
    test_scores.to_csv(f"{save_dir}/test_scores_sub_{sub}.csv", index=False)
    inner_cv_results.to_csv(f"{save_dir}/inner_cv_results_sub_{sub}.csv", index=False)
    joblib.dump({"model": best_model, "params": params}, f"{save_dir}/best_model_sub_{sub}.joblib")
