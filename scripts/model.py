import numpy as np



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from helpers import load_data, individual_z_scoring, training_splits
    
def run_regression(X, Y, train_split, val_split, test_split):
    pass

def run_nn(X, Y, train_split, val_split, test_split):
    pass

def run_svm(X, Y, train_split, val_split, test_split):
    pass

def run_forest(X, Y, train_split, val_split, test_split):
    pass

def run_log_reg(X, Y, train_split, val_split, test_split):
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
    masks = ["brain_masked", "reward_masked"]
    maps = ['lss_maps', 'lss_t_maps', 'lss_z_maps']
    splits = ["subs", "tasks", "sub_task"]

    map = maps[0]
    mask = masks[0]

    X, Y, index_ranges = load_data(tasks, mask, map)

    X = individual_z_scoring(X, index_ranges)
    Y = np.sign(Y)

    train, val, test = training_splits(index_ranges, split=splits[0])

    run_log_reg(X, Y, train, val, test)
    # print(Y.shape)
    # print(f"Train: {len(train)} {len(train[0])}\n\n")
    # print(f"Val: {len(val)} {len(val[0])}\n\n")
    # print(f"Test: {len(test)}\n\n")

