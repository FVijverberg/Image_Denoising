import numpy as np
import matplotlib.pyplot as plt

# Load preprocessed patches
Y = np.load("../Data/Y_patches.npy")   # 64 x 6000
n_features, n_samples = Y.shape
n_atoms = 256
n_iter = 300
seed = 0
rng = np.random.default_rng(seed)

# Center data
patch_means = np.mean(Y, axis=0, keepdims=True)
Y_centered = Y - patch_means

# Initialize dictionary, hyperparameters, and residual variance
indices = rng.choice(n_samples, size=n_atoms, replace=False) 
D = Y_centered[:, indices].copy()
D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-12

gamma = np.ones(n_atoms)
sigma2 = np.var(Y_centered)  # Initial data-driven variance guess

# DL-SBL EM Core Loop
for it in range(n_iter):
    # --- E-Step ---
    Inv_Gamma = np.diag(1.0 / (gamma + 1e-12))
    Sigma = np.linalg.pinv(
    (D.T @ D) / sigma2 + Inv_Gamma
    )
    M = (1.0 / sigma2) * (Sigma @ D.T @ Y_centered)  # Posterior means matrix

    # --- M-Step ---
    gamma = np.mean(M**2, axis=1) + np.diag(Sigma)
    
    # Dictionary Update
    D = Y_centered @ M.T @ np.linalg.inv(
        n_samples * Sigma + M @ M.T
    )

    norms = np.linalg.norm(D, axis=0) + 1e-12
    D /= norms
    gamma *= norms**2
    # Update sigma^2 (Option 2: Automated estimation from residual energy)
    tr_term = n_samples * np.sum((D @ Sigma) * D)
    sigma2 = (np.sum((Y_centered - D @ M) ** 2) + tr_term) / (n_features * n_samples)

    mse = np.mean((Y_centered - D @ M) ** 2)
    print(f"DL-SBL Iteration {it + 1}/{n_iter} | Reconstruction MSE: {mse:.4f} | Estimated sigma2: {sigma2:.4f}")

# Save results
np.save("../Data/dictionary_D.npy", D)
np.save("../Data/sparse_X.npy", M)  # Save posterior mean as X for cross-compatibility
np.save("../Data/gamma.npy", gamma)
np.save("../Data/patch_means.npy", patch_means)
np.save("../Data/sigma2.npy", sigma2)
print("\nSaved DL-SBL artifacts to ../Data/")

# --- Visualizations ---
fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(D[:, i].reshape(8, 8), cmap="gray")
    ax.axis("off")
plt.suptitle("First 64 DL-SBL Dictionary Atoms")
plt.tight_layout()
plt.show()