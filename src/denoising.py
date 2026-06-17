# Denoise images using a learned dictionary K-SVD/OMP and DL-SBL.
# Based on the Elad & Aharon (2006) patch-based denoising approach.

import time
import numpy as np
from sklearn.linear_model import orthogonal_mp
import utils
from dict_learning_dlsbl import sbl_recover

patch = 8# 8x8 pixels

def extract_patches(img, stride):
     # split img up into patches
    h, w = img.shape # total height and width

    # Get all possible top-left pos.
    rows = list(range(0, h - patch + 1, stride)) + [h - patch] #2nd term ensures bottom included
    cols = list(range(0, w - patch + 1, stride)) + [w - patch]

    positions = [(i, j) for i in dict.fromkeys(rows) for j in dict.fromkeys(cols)] #dict.fromkeys used in case edge pos. already in list
    Y = np.stack([img[i:i + patch, j:j + patch].ravel() for i, j in positions], axis=1) # extract patches and stack as col
    return Y, positions, (h, w) #Return patch matrix (64, P), top-left positions, and image shape.


def assemble(patches_hat, positions, shape):
    # Avg overlapping reconstructed patches back into an image.
    accum = np.zeros(shape)
    weight = np.zeros(shape) #to count number of patches covering ea h pixel

    for p, (i, j) in enumerate(positions):
        accum[i:i + patch, j:j + patch] += patches_hat[:, p].reshape(patch, patch) #add patch to accum
        weight[i:i + patch, j:j + patch] += 1.0 #increment
    return accum / weight # divide by count


def denoise_image(noisy_img, D, method, sigma=None, stride=4):
    #Denoise an image. Returns (denoised_image, avg_atoms_per_patch).
    Y, positions, shape = extract_patches(noisy_img, stride)
    means = Y.mean(axis=0, keepdims=True)
    Yc = Y - means  # remove dc

    if method == "omp":
        epsilon = sigma * np.sqrt(patch * patch)  # error bound
        X = orthogonal_mp(D,  Yc, tol= epsilon ** 2)
    else:
        X = sbl_recover(Yc, D, sigma**2)

    patches_hat = D @ X + means  # reconstruct and add DC back
    denoised = assemble(patches_hat, positions, shape) #avg overlapping patches


    # SBL codes never exactly zero, so threshold at 1% of the max to count active atoms
    thr = 0.01 * np.max(np.abs(X), axis=0, keepdims=True) + 1e-12
    avg_atoms = np.mean(np.sum(np.abs(X) > thr, axis=0))# active
    return np.clip(denoised, 0, 255), avg_atoms #Return denoised_image, avg_atoms_per_patch


if __name__ == "__main__":
    method = "omp"  # or sbl

    if method == "omp":
        D = np.load("../Data/dictionary_D_omp.npy") # load dict

    test_files = sorted((utils.DATA_ROOT / "test").glob("*.jpg")) #ids
    rng = np.random.default_rng(42)
    selected_files = rng.choice(test_files, size=3, replace=False) # choose 3

    results = []
    for file in selected_files:
        clean = utils.load_image_gray(file)
        img_seed = utils.get_image_seed(file.stem) 
        for sigma in [5, 10, 15, 25]:
            if method == "sbl":
                D = np.load(f"../Data/dictionary_D_sbl_{sigma}.npy")  # sigma dependant dictionary
            noise_rng = np.random.default_rng(img_seed * 1000 + sigma) # different images get diff. noise
            noisy = np.clip(clean + noise_rng.normal(scale=sigma, size=clean.shape), 0, 255)

            t0 = time.time()
            denoised, atoms = denoise_image(noisy, D, method, sigma)
            elapsed = time.time() - t0

            results.append([
                file.stem, sigma,
                utils.mse(clean, noisy), utils.mse(clean, denoised),
                utils.psnr(clean, noisy), utils.psnr(clean, denoised),
                sigma * np.sqrt(64) if method == "omp" else 0.0, atoms, elapsed,
            ])
            print(f"{file.stem} | sigma={sigma:2d} | "
                  f"PSNR {utils.psnr(clean, noisy):.2f} -> {utils.psnr(clean, denoised):.2f} dB"
                  f" | {elapsed:.1f}s")

    utils.save_metrics(results, method)