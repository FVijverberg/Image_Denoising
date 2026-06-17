# dictionary_learning.py
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import orthogonal_mp

# Load  patches
Y = np.load("../Data/Y_patches.npy") # from preprocessing
n_features, n_samples = Y.shape
n_atoms = 128
epsilon = 20 * np.sqrt(64)   # sigma * sqrt(m)
n_iter = 20

rng = np.random.default_rng(42)

patch_means = np.mean(Y, axis=0, keepdims=True) #dc
Y_centered = Y - patch_means # remove mean

# Initialize dictionary with random training patches
indices = rng.choice(n_samples, size=n_atoms, replace=False)
D = Y_centered[:, indices].copy()
D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-12

# K-SVD Optimization Loop
t0 = time.time()
for it in range(n_iter):
    print(f"K-SVD iteration {it + 1}/{n_iter}")

    X = orthogonal_mp(D, Y_centered, tol=epsilon**2, precompute=True) # sparse coding step

    # Dictionary and Code Update Step
    for k in range(n_atoms):
        omega = np.nonzero(X[k, :])[0]

        if len(omega) == 0:  # Dead atom replacement
            random_patch = rng.integers(n_samples)
            D[:, k] = Y_centered[:, random_patch]
            D[:, k] /= np.linalg.norm(D[:, k]) + 1e-12
            continue

        X[k, omega] = 0  # Temporarily isolate atom contribution
        E_k = Y_centered[:, omega] - D @ X[:, omega]

        # SVD Rank-1 approximation update
        U, S, Vt = np.linalg.svd(E_k, full_matrices=False)
        D[:, k] = U[:, 0]
        X[k, omega] = S[0] * Vt[0, :]

    mse_loss = np.mean((Y_centered - D @ X) ** 2)
    print(f"   Reconstruction MSE: {mse_loss:.4f}")

elapsed = time.time() - t0
print(f"\nK-SVD training time: {elapsed:.1f} s ({elapsed/60:.1f} min)")

# Save K-SVD artifacts
np.save("../Data/dictionary_D_omp.npy", D)
np.save("../Data/sparse_X_omp.npy", X)
np.save("../Data/patch_means_omp.npy", patch_means)
print("Saved K-SVD artifacts to ../Data/*_omp.npy")