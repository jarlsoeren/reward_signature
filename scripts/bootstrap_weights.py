import os
import argparse
import numpy as np
import joblib

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True,
                    choices=["gonogo", "mid", "risksensitive", "hcp", "twostep"])
parser.add_argument("--idx", type=int, required=True,
                    help="Bootstrap job index (0–9)")
args = parser.parse_args()

TASK = args.task
IDX = args.idx

# -----------------------------
# Bootstrap config
# -----------------------------
N_BOOTSTRAPS = 100

print(f"Task={TASK}, idx={IDX}")

task_dir = f"/mnt/scratch/projects/rewardMap/reward_signature/final_model/{TASK}"

data = np.load(os.path.join(task_dir, "data.npz"), allow_pickle=True)
X = data["X"]
Y_signed = data["Y_signed"]

model = joblib.load(os.path.join(task_dir, "best_model.joblib"))
model["model"].n_jobs = -1


rng = np.random.default_rng(seed=IDX)  # reproducible per job

all_weights = []

n = X.shape[0]
n_voxels = X.shape[1]

for b in range(N_BOOTSTRAPS):

    bootstrap_idx = rng.choice(n, size=n, replace=True)

    X_boot = X[bootstrap_idx]
    Y_boot = Y_signed[bootstrap_idx]

    model.fit(X_boot, Y_boot)

    weights_pc = model["model"].coef_
    weights_original = model["pca"].inverse_transform(weights_pc)

    selected_idx = model["correlate"].selected_idx_

    assert weights_original.shape[1] == len(selected_idx), \
        "Mismatch: weights_original must match number of selected voxels."

    if TASK in ("risksensitive", "twostep"):
        full_weights = np.zeros(n_voxels)
        full_weights[selected_idx] = weights_original
    else:
        full_weights = np.zeros((weights_original.shape[0], n_voxels))
        full_weights[:, selected_idx] = weights_original

    all_weights.append(full_weights)

all_weights = np.stack(all_weights, axis=0)

print("Partial array shape:", all_weights.shape)

# -----------------------------
# Save per-job output
# -----------------------------
out_file = os.path.join(
    task_dir, f"bootstrap_weights_idx{IDX:02d}.npy"
)
np.save(out_file, all_weights)

print("Saved:", out_file)
