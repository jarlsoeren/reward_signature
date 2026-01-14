import os
import json
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, make_scorer
from sklearn.model_selection import ParameterSampler

PARAMS = [
    {
        "correlate__keep_ratio": [0.1, 0.4, 0.7, 1.0],
        "pca__n_components": [10, 25, 50, 75],
        "model__penalty": ["l1", "l2"],
        "model__C": [0.01, 0.1, 1.0, 10.0]
    },
]

param_list = list(ParameterSampler(PARAMS, n_iter=50, random_state=42))
tasks = ['risksensitive', 'twostep', 'hcp', 'mid', 'gonogo']

out_dir = "job_params"
os.makedirs(out_dir, exist_ok=True)

job_index = 0
job_map = []

for task in tasks:
    for sub in range(15):
        fname = f"{out_dir}/job_{job_index}.json"
        with open(fname, "w") as f:
            json.dump({
                "task": task,
                "sub": sub,
                "params": param_list
            }, f, indent=2)
        job_map.append(fname)
        job_index += 1

print("Total jobs:", len(job_map))
