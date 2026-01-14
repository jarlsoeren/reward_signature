import numpy as np
import pandas as pd
import os
import joblib
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import balanced_accuracy_score, f1_score, make_scorer
from correlateTransformer import CorrelateTransformer

from helpers import load_data, training_splits


CLASSIFICATION_SCORING = {   
    "balanced_acc": make_scorer(balanced_accuracy_score),
    "macro_f1": make_scorer(f1_score, average="macro")
}


PARAMS = {
        "correlate__keep_ratio": [0.1, 0.4, 0.7, 1.0],
        "pca__n_components": [10, 25, 50, 75],
        "model__penalty": ["l1", "l2"],
        "model__C": [0.01, 0.1, 1.0, 10.0]
    }


def run_within_task_cv(X, Y, n_splits=4, n_iter=50):


    pipe = Pipeline([
        ("scaler", StandardScaler()), 
        ("correlate", CorrelateTransformer()),
        ("pca", PCA()), 
        ("model", LogisticRegression(solver="saga", max_iter=10000, class_weight="balanced"))  # parameters get replaced
    ])
    

    # Filter parameters
    refit_metric = list(CLASSIFICATION_SCORING.keys())[1]


    # Randomized search
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=PARAMS,
        n_iter=n_iter,
        scoring=CLASSIFICATION_SCORING,
        refit=refit_metric,
        cv=n_splits,
        verbose=4,
        n_jobs=-1,
        random_state=42,
        return_train_score=True
    )

    search.fit(X, Y)

    # Extract cv_results_
    cv_results = pd.DataFrame(search.cv_results_)
    
    return search.best_estimator_, search.best_score_, cv_results


if __name__ == "__main__":

    SAVE_DIR = "/mnt/scratch/projects/rewardMap/reward_signature/final_model"
    os.makedirs(SAVE_DIR, exist_ok=True)

    task = 'gonogo'
    mask = "cut_brain_masked"
    map = "lss_z_maps"

    n_splits = 4

    print(f"\n===== Running within-task CV for: {task} =====")
    X, Y, trial_types, index_ranges = load_data([task], mask, map)
    #train_splits, sub_splits, _ = training_splits(index_ranges, split="subs") # all subs held out once

    rng = np.random.default_rng(seed=42)

    Y_signed = np.sign(Y)


    best_model, best_score, cv_results = run_within_task_cv(X, Y_signed, n_splits=4, n_iter=50) 

    task_dir = os.path.join(SAVE_DIR, task)
    os.makedirs(task_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(task_dir, "data.npz"),
        X=X,
        Y=Y,
        Y_signed=Y_signed,
        trial_types=np.array(trial_types),
        index_ranges=np.array(index_ranges, dtype=object),
        #sub_splits=np.array(sub_splits, dtype=object),
    )

    joblib.dump(best_model, os.path.join(task_dir, "best_model.joblib"))

    with open(os.path.join(task_dir, "best_score.json"), "w") as f:
        json.dump({"best_score": best_score}, f, indent=2)

    cv_results.to_csv(os.path.join(task_dir, "cv_results.csv"), index=False)

    print(f"\nSaved outputs to: {task_dir}")


