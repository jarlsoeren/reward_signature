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
from sklearn.metrics import balanced_accuracy_score, f1_score, make_scorer, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from correlateTransformer import CorrelateTransformer

from helpers import load_data, individual_z_scoring

# ====================================================
# ---- Model Configurations ----
# ====================================================

REGRESSION_SCORING = {
    "mean_absolute_error": make_scorer(mean_absolute_error),
    "r2_score": make_scorer(r2_score),
}


MODEL_CONFIGS = {
    "regression": {
        "estimator": LinearRegression,
        "params": {
            "pca__n_components": [50],
            "correlate__keep_ratio": [0.1, 0.5, 1.0],
            "pca__whiten": [False],
        },
        "scoring": REGRESSION_SCORING
    },
}


# ====================================================
# ---- Run Within-Task 4-Fold CV ----
# ====================================================
def run_within_task_cv(X, Y, n_splits=4, n_iter=10, use_pca=True):
    cfg = MODEL_CONFIGS["regression"]

    # Build pipeline
    steps = [("scaler", StandardScaler())]
    steps.append(("correlate", CorrelateTransformer()))
    steps.append(("pca", PCA()))
    steps.append(("model", LinearRegression()))
    pipe = Pipeline(steps)

    # Filter parameters
    param_grid = {k: v for k, v in cfg["params"].items() if use_pca or not k.startswith("pca")}
    scoring = cfg["scoring"]
    refit_metric = list(scoring.keys())[1]

    # KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Randomized search
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        refit=refit_metric,
        cv=kf,
        verbose=2,
        n_jobs=1,
        random_state=42,
        return_train_score=True
    )

    search.fit(X, Y)

    print("\n===== Best Hyperparameters =====")
    print(search.best_params_)
    print(f"Best CV {refit_metric}: {search.best_score_:.4f}")

    # Extract cv_results_
    cv_results = search.cv_results_
    n_param_sets = len(cv_results["params"])
    n_folds = kf.get_n_splits()

    all_scores = {}

    print("\n===== All Hyperparameter Sets =====")
    for i in range(n_param_sets):
        params = cv_results["params"][i]
        print(f"\nHyperparameter set {i}: {params}")

        metrics = ["r2_score", "mean_absolute_error"]
        metric_scores = {}

        for metric in metrics:
            # Aggregate metrics
            mean_train = cv_results[f"mean_train_{metric}"][i]
            std_train = cv_results[f"std_train_{metric}"][i]
            mean_val = cv_results[f"mean_test_{metric}"][i]
            std_val = cv_results[f"std_test_{metric}"][i]

            print(f"  [{metric}]")
            print(f"    Mean train: {mean_train:.4f} ± {std_train:.4f}")
            print(f"    Mean val:   {mean_val:.4f} ± {std_val:.4f}")

            # Per-fold scores
            fold_scores = []
            for fold in range(n_folds):
                train_score = cv_results[f"split{fold}_train_{metric}"][i]
                val_score = cv_results[f"split{fold}_test_{metric}"][i]
                fold_scores.append({"train": train_score, "val": val_score})
                print(f"      Fold {fold+1}: Train = {train_score:.4f}, Val = {val_score:.4f}")

            metric_scores[metric] = {
                "mean_train": mean_train,
                "std_train": std_train,
                "mean_val": mean_val,
                "std_val": std_val,
                "folds": fold_scores
            }

        all_scores[i] = {"params": params, "metrics": metric_scores}

    return search.best_estimator_, search.best_score_, all_scores, kf


if __name__ == "__main__":

    SAVE_DIR = "/mnt/scratch/projects/rewardMap/reward_signature/model_outputs_reg_corr"
    os.makedirs(SAVE_DIR, exist_ok=True)

    tasks = ['gonogo', 'hcp', 'mid', 'risksensitive', 'twostep']

    mask = "brain_masked"
    map = "lss_maps"
    individual_z = False
    use_pca = True

    for task in tasks:
        print(f"\n===== Running within-task CV for: {task} =====")
        X, Y, _ = load_data([task], mask, map)

        if individual_z:
            X = individual_z_scoring(X, None)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42, shuffle=True)

        best_model, best_score, all_scores, kf = run_within_task_cv(X_train, y_train, n_iter=3, n_splits=4, use_pca=use_pca)

        # 4️ Save data splits and model for later error digging
        task_dir = os.path.join(SAVE_DIR, task)
        os.makedirs(task_dir, exist_ok=True)

        np.savez_compressed(
            os.path.join(task_dir, "data_splits.npz"),
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test
        )

        joblib.dump(best_model, os.path.join(task_dir, "best_model.joblib"))
        joblib.dump(kf, os.path.join(task_dir, "kf.pkl"))
        
        df = pd.DataFrame.from_dict(all_scores)
        df.to_csv(os.path.join(task_dir, "all_scores.csv"), index=False)

