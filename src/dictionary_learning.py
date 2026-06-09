import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import orthogonal_mp


# -----------------------------
# Load patches
# -----------------------------
Y = np.load("../Data/Y_patches.npy")   # shape: (64, 6000)

n_features, n_samples = Y.shape
n_atoms = 256
sparsity = 6          # T0: max atoms used per patch
n_iter = 20
seed = 0

rng = np.random.default_rng(seed)

# -----------------------------
# Normalize / center patches
# -----------------------------
patch_means = np.mean(Y, axis=0, keepdims=True)
Y_centered = Y - patch_means


# -----------------------------
# Initialize dictionary D
# pick random training patches as initial atoms
# -----------------------------
indices = rng.choice(n_samples, size=n_atoms, replace=False)
D = Y_centered[:, indices].copy()

D_norms = np.linalg.norm(D, axis=0, keepdims=True) + 1e-12
D = D / D_norms


# -----------------------------
# K-SVD
# -----------------------------
for it in range(n_iter):
    print(f"\nK-SVD iteration {it + 1}/{n_iter}")

    # Step 1: Sparse coding using OMP
    # Solves: Y_centered ≈ D @ X
    X = orthogonal_mp(D, Y_centered, n_nonzero_coefs=sparsity)

    # Step 2: Dictionary update
    for k in range(n_atoms):

        # Find patches that use atom k
        omega = np.nonzero(X[k, :])[0]

        # If no patch uses this atom, reinitialize it
        if len(omega) == 0:
            random_patch = rng.integers(n_samples)
            D[:, k] = Y_centered[:, random_patch]
            D[:, k] /= np.linalg.norm(D[:, k]) + 1e-12
            continue

        # Remove contribution of atom k
        X_k = X[k, omega].copy()
        X[k, omega] = 0

        # Error restricted to patches using atom k
        E_k = Y_centered[:, omega] - D @ X[:, omega]

        # Rank-1 SVD update
        U, S, Vt = np.linalg.svd(E_k, full_matrices=False)

        # Update atom
        D[:, k] = U[:, 0]

        # Update coefficients for atom k
        X[k, omega] = S[0] * Vt[0, :]

    # Reconstruction error after this iteration
    Y_hat = D @ X
    mse = np.mean((Y_centered - Y_hat) ** 2)
    print(f"Reconstruction MSE: {mse:.4f}")


# Save results
np.save("../Data/dictionary_D.npy", D)
np.save("../Data/sparse_X.npy", X)
np.save("../Data/patch_means.npy", patch_means)

print("\nSaved:")
print("../Data/dictionary_D.npy")
print("../Data/sparse_X.npy")
print("../Data/patch_means.npy")


# -----------------------------
# Sanity checks
# -----------------------------
print("\n========== SANITY CHECKS ==========")

print("\n----- Shape Check -----")
print("Y shape:", Y.shape)
print("Y_centered shape:", Y_centered.shape)
print("D shape:", D.shape)
print("X shape:", X.shape)

print("\nExpected:")
print("Y: (64, 6000)")
print("D: (64, 256)")
print("X: (256, 6000)")

print("\n----- NaN / Inf Check -----")
print("NaNs in D:", np.isnan(D).sum())
print("NaNs in X:", np.isnan(X).sum())
print("Infs in D:", np.isinf(D).sum())
print("Infs in X:", np.isinf(X).sum())

print("\n----- Dictionary Statistics -----")
print("D min:", D.min())
print("D max:", D.max())
print("D mean:", D.mean())
print("D std:", D.std())

print("\n----- Atom Norm Check -----")
atom_norms = np.linalg.norm(D, axis=0)
print("Min atom norm:", atom_norms.min())
print("Max atom norm:", atom_norms.max())
print("Mean atom norm:", atom_norms.mean())

print("\n----- Sparsity Check -----")
threshold = 1e-8
nonzero = np.sum(np.abs(X) > threshold)
total = X.size
print("Total coefficients:", total)
print("Nonzero coefficients:", nonzero)
print("Percentage nonzero:", 100 * nonzero / total, "%")
print("Average nonzeros per patch:", nonzero / n_samples)

print("\n----- Reconstruction Check -----")
Y_hat_centered = D @ X
Y_hat = Y_hat_centered + patch_means

mse_centered = np.mean((Y_centered - Y_hat_centered) ** 2)
mse_original = np.mean((Y - Y_hat) ** 2)

print("Centered reconstruction MSE:", mse_centered)
print("Original-scale reconstruction MSE:", mse_original)

print("===================================")


# -----------------------------
# Visualize dictionary atoms
# -----------------------------
fig, axes = plt.subplots(8, 8, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    atom = D[:, i].reshape(8, 8)
    ax.imshow(atom, cmap="gray")
    ax.axis("off")

plt.suptitle("First 64 K-SVD Dictionary Atoms")
plt.tight_layout()
plt.show()


# -----------------------------
# Visualize one patch reconstruction
# -----------------------------
idx = rng.integers(n_samples)

original_patch = Y[:, idx].reshape(8, 8)
reconstructed_patch = Y_hat[:, idx].reshape(8, 8)

fig, axes = plt.subplots(1, 2, figsize=(6, 3))

axes[0].imshow(original_patch, cmap="gray")
axes[0].set_title("Original Patch")
axes[0].axis("off")

axes[1].imshow(reconstructed_patch, cmap="gray")
axes[1].set_title("K-SVD Reconstruction")
axes[1].axis("off")

plt.tight_layout()
plt.show()