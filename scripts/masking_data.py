import os
import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib.pyplot as plt

from nilearn.image import concat_imgs, resample_to_img
from nilearn.masking import apply_mask
from nilearn.datasets import fetch_atlas_pauli_2017

def cut_mask(in_file):
    nii = nib.load(in_file)
    data = nii.get_fdata()
    affine = nii.affine
    header = nii.header
    trimmed = data[:, :, 12:]    # drop slices 0–11
    trimmed = np.concatenate((np.zeros_like(data[:, :, :12]), trimmed), axis=2)  # just to show original vs trimmed
    trimmed_nii = nib.Nifti1Image(trimmed, affine, header)

    return trimmed_nii

def brain_cut_mask(maps_dir, mask_path, out_dir, tasks):
    """
    Applies a standard brain mask to LSS maps and saves masked beta/reward arrays.
    """
    mask = cut_mask(mask_path)
    for task in tasks:
        p = os.path.join(maps_dir, task)
        subject_dirs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]

        out_p = os.path.join(out_dir, task)
        os.makedirs(out_p, exist_ok=True)

        for i, sub in enumerate(subject_dirs):
            sub_dir_p = os.path.join(p, sub)

            img_files, rewards_file = [], None
            for root, dirs, files in os.walk(sub_dir_p):
                for file in files:
                    if file.endswith(".nii.gz"):
                        img_files.append(file)
                    elif file.startswith("rewards"):
                        rewards_file = os.path.join(sub_dir_p, file)

            # Load and concatenate subject images
            imgs = [nib.load(os.path.join(sub_dir_p, f)) for f in sorted(img_files)]
            imgs = concat_imgs(imgs)

            # Apply brain mask
            masked_data = apply_mask(imgs=imgs, mask_img=mask)

            # Load rewards
            if rewards_file is not None:
                rewards = pd.read_csv(rewards_file)["reward"].to_numpy()
            else:
                rewards = np.array([])

            # Save output
            np.save(os.path.join(out_p, f"betas_sub_{i:03d}.npy"), masked_data)
            np.save(os.path.join(out_p, f"rewards_sub_{i:03d}.npy"), rewards)

            print(f"[brain_mask] Saved masked data for {task}/{sub}")


def reward_mask(maps_dir, out_dir, tasks, n_subs=None):
    """
    Applies a reward-related subcortical mask (Putamen, Caudate, Nucleus Accumbens)
    from Pauli et al. 2017 to LSS maps and saves masked beta/reward arrays.
    """
    # Fetch and prepare Pauli atlas
    atlas = fetch_atlas_pauli_2017(atlas_type="deterministic")
    img = nib.load(atlas.maps)
    data = img.get_fdata()

    # Define reward-related regions
    putamen_mask = np.isin(data, 1).astype(np.int8)
    caudate_mask = np.isin(data, 2).astype(np.int8)
    nucleus_accumbens_mask = np.isin(data, 3).astype(np.int8)

    combined_mask = putamen_mask + caudate_mask + nucleus_accumbens_mask
    combined_mask_img = nib.Nifti1Image(combined_mask, img.affine, img.header)

    for task in tasks:
        p = os.path.join(maps_dir, task)
        subject_dirs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]

        out_p = os.path.join(out_dir, task)
        os.makedirs(out_p, exist_ok=True)

        for i, sub in enumerate(subject_dirs):
            if n_subs is not None and i >= n_subs:
                break

            sub_dir_p = os.path.join(p, sub)

            img_files, rewards_file = [], None
            for root, dirs, files in os.walk(sub_dir_p):
                for file in files:
                    if file.endswith(".nii.gz"):
                        img_files.append(file)
                    elif file.startswith("rewards"):
                        rewards_file = os.path.join(sub_dir_p, file)

            imgs = [nib.load(os.path.join(sub_dir_p, f)) for f in sorted(img_files)]
            imgs = concat_imgs(imgs)

            # Resample reward mask to subject data space
            mask_resampled = resample_to_img(
                combined_mask_img, imgs, interpolation='nearest', 
                force_resample=True, copy_header=True
            )

            # Apply reward mask
            masked_data = apply_mask(imgs=imgs, mask_img=mask_resampled)

            # Load rewards
            if rewards_file is not None:
                rewards = pd.read_csv(rewards_file)["reward"].to_numpy()
            else:
                rewards = np.array([])

            # Save output
            np.save(os.path.join(out_p, f"betas_sub_{i:03d}.npy"), masked_data)
            np.save(os.path.join(out_p, f"rewards_sub_{i:03d}.npy"), rewards)

            print(f"[reward_mask] Saved masked data for {task}/{sub}")


def brain_mask(maps_dir, mask_path, out_dir, tasks):
    """
    Applies a standard brain mask to LSS maps and saves masked beta/reward arrays.
    """
    for task in tasks:
        p = os.path.join(maps_dir, task)
        subject_dirs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]

        out_p = os.path.join(out_dir, task)
        os.makedirs(out_p, exist_ok=True)

        for i, sub in enumerate(subject_dirs):
            sub_dir_p = os.path.join(p, sub)

            img_files, rewards_file = [], None
            for root, dirs, files in os.walk(sub_dir_p):
                for file in files:
                    if file.endswith(".nii.gz"):
                        img_files.append(file)
                    elif file.startswith("rewards"):
                        rewards_file = os.path.join(sub_dir_p, file)

            # Load and concatenate subject images
            imgs = [nib.load(os.path.join(sub_dir_p, f)) for f in sorted(img_files)]
            imgs = concat_imgs(imgs)

            # Apply brain mask
            masked_data = apply_mask(imgs=imgs, mask_img=mask_path)

            # Load rewards
            if rewards_file is not None:
                rewards = pd.read_csv(rewards_file)["reward"].to_numpy()
            else:
                rewards = np.array([])

            # Save output
            np.save(os.path.join(out_p, f"betas_sub_{i:03d}.npy"), masked_data)
            np.save(os.path.join(out_p, f"rewards_sub_{i:03d}.npy"), rewards)

            print(f"[brain_mask] Saved masked data for {task}/{sub}") 

# =========================
# --- Example Usage -------
# =========================

if __name__ == "__main__":
    maps_dir = '/mnt/projects/rewardMap/STUDIES/pilotstudy/derivatives/lss_z_maps/'
    tasks = ['gonogo', 'hcp', 'mid', 'risksensitive', 'twostep']

    brain_mask_path = "/mnt/projects/rewardMap/STUDIES/pilotstudy/derivatives/masks/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
    
    brain_out_dir = "/mnt/scratch/projects/rewardMap/reward_signature/brain_masked/lss_z_maps"
    reward_out_dir = "/mnt/scratch/projects/rewardMap/reward_signature/reward_masked/lss_z_maps"
    cut_out_dir = "/mnt/scratch/projects/rewardMap/reward_signature/cut_brain_masked/lss_z_maps"

    # Run both masking procedures
    #brain_mask(maps_dir, brain_mask_path, brain_out_dir, tasks)
    #reward_mask(maps_dir, reward_out_dir, tasks, n_subs=None)
    brain_cut_mask(maps_dir, brain_mask_path, cut_out_dir, tasks)
