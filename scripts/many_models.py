import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score, f1_score, make_scorer

from helpers import load_data, individual_z_scoring, training_splits

# ====================================================
# ---- Model Configurations ----
# ====================================================

CLASSIFICATION_SCORING = {
    "balanced_acc": make_scorer(balanced_accuracy_score),
    "macro_f1": make_scorer(f1_score, average="macro")
}

REGRESSION_SCORING = {
    
}

MODEL_CONFIGS = {
    "regression": {
        "estimator": LinearRegression,
        "params": {"pca__n_components": [10, 25, 50], "pca__whiten": [True, False]},
        "scoring": None
    },
    "nn": {
        "estimator": MLPClassifier,
        "params": {
            "pca__n_components": [25, 50],
            "model__hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "model__activation": ["relu", "tanh"],
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__learning_rate_init": [0.001, 0.01],
            "model__max_iter": [500, 1000]
        },
        "scoring": CLASSIFICATION_SCORING
    },
    "svm": {
        "estimator": SVC,
        "params": {
            "pca__n_components": [10, 25, 50],
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"]
        },
        "scoring": CLASSIFICATION_SCORING
    },
    "forest": {
        "estimator": RandomForestClassifier,
        "params": {
            "pca__n_components": [10, 25, 50],
            "model__n_estimators": [100, 200, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4]
        },
        "scoring": CLASSIFICATION_SCORING
    },
    "logreg": {
        "estimator": LogisticRegression,
        "params": {
            "pca__n_components": [10, 25, 50],
            "pca__whiten": [True, False],
            "model__C": [0.1, 1, 10],
            "model__max_iter": [1000, 5000, 10000],
            "model__penalty": [None, "l1", "l2"]
        },
        "scoring": CLASSIFICATION_SCORING
    }
}

# ====================================================
# ---- Base Training Utility ----
# ====================================================
def evaluate_model(search, X_test, y_test):
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = balanced_accuracy_score(y_test, y_pred)

    for mean_train, mean_val, params in zip(
        search.cv_results_['mean_train_score'],
        search.cv_results_['mean_test_score'],
        search.cv_results_['params']
    ):
        print(f"{params} -> Train: {mean_train:.4f}, Val: {mean_val:.4f}")

    print("Best params:", search.best_params_)
    print("Validation score:", search.best_score_)
    print("Test balanced accuracy:", acc)
    return best_model, acc

# ====================================================
# ---- Unified Model Runner with Multi-Scoring ----
# ====================================================
def run_model_cv(model_name, X, Y, train_split, val_split, test_split,
                 n_iter=10, n_splits=5, use_pca=True):
    
    cfg = MODEL_CONFIGS[model_name]
    X_test, y_test = X[test_split], Y[test_split]

    steps = [("scaler", StandardScaler())] if model_name != "regression" else []
    if use_pca:
        steps.append(("pca", PCA()))
    steps.append(("model", cfg["estimator"](random_state=42) if model_name != "regression" else cfg["estimator"]()))

    pipe = Pipeline(steps)
    cv_splits = list(zip(train_split, val_split))[:n_splits]

    param_grid = {k: v for k, v in cfg["params"].items() if use_pca or not k.startswith("pca")}

    # Determine scoring
    scoring = cfg["scoring"]
    refit_metric = None
    if isinstance(scoring, dict):
        refit_metric = "balanced_acc"  # choose metric to pick best model
    else:
        refit_metric = True  # default for regression or single metric

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        refit=refit_metric,
        cv=cv_splits,
        verbose=2,
        n_jobs=1,
        random_state=42,
        return_train_score=True
    )

    search.fit(X, Y)

    if model_name == "regression":
        print("Best params:", search.best_params_)
        print("Validation R^2:", search.best_score_)
        return search.best_estimator_
    else:
        return evaluate_model(search, X_test, y_test)


# ====================================================
# ---- MAIN ----
# ====================================================
if __name__ == "__main__":
    tasks = ['gonogo', 'hcp', 'mid', 'risksensitive', 'twostep']

    masks = ["brain_masked", "reward_masked"]
    maps = ['lss_maps', 'lss_t_maps', 'lss_z_maps']
    splits = ["subs", "tasks", "sub_task"]
    models = ['logreg', 'svm', 'forest', 'regression', 'nn']

    mask = masks[0]
    map = maps[0]
    split = splits[0]
    model = models[1]  # choose model
    individual_z = True
    use_pca = False

    X, Y, index_ranges = load_data(tasks, mask, map)
    if individual_z:
        X = individual_z_scoring(X, index_ranges)

    if model != 'regression':
        Y = np.sign(Y)

    train, val, test = training_splits(index_ranges, split=split)

    run_model_cv(model, X, Y, train, val, test, n_iter=10, n_splits=5, use_pca=use_pca)
