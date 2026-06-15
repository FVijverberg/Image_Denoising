import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io
from skimage.color import rgb2gray

Data_root = Path("../Data/images")
test_dir = Data_root / "test"
output_dir = Path("../Data/denoised_results")
output_dir.mkdir(exist_ok=True)

sigmas = [5, 10, 15, 25]
patch_size = 8
seed = 0
rng = np.random.default_rng(seed)

# Load trained DL-SBL items
D = np.load("../Data/dictionary_D.npy")
gamma = np.load("../Data/gamma.npy")
print("Loaded DL-SBL Dictionary D:", D.shape)

def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def psnr(clean, estimate):
    error = mse(clean, estimate)
    if error == 0:
        return float("inf")
    return 10 * np.log10((255.0 ** 2) / error)

def denoise_image_sbl(noisy_img, D, gamma, sigma_n, patch_size=8, stride=1):
    h, w = noisy_img.shape

    accum  = np.zeros((h, w), dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)

    # Precompute the Bayesian MMSE reconstruction matrix for this noise variance
    n_features = patch_size ** 2
    Gamma = np.diag(gamma)
    W = Gamma @ D.T @ np.linalg.inv((sigma_n**2) * np.eye(n_features) + D @ Gamma @ D.T)
    R = D @ W

    h_max = h - patch_size + 1
    w_max = w - patch_size + 1

    def positions(max_pos, step):
        pos = list(range(0, max_pos, step))
        if pos[-1] != max_pos - 1:   # ensure the last row/col is always covered
            pos.append(max_pos - 1)
        return pos

    for i in positions(h_max, stride):
        for j in positions(w_max, stride):
            patch = noisy_img[i:i+patch_size, j:j+patch_size]

            patch_mean = np.mean(patch)
            patch_centered = (patch - patch_mean).flatten()

            patch_hat = R @ patch_centered
            patch_hat = patch_hat.reshape(patch_size, patch_size) + patch_mean

            accum[i:i+patch_size, j:j+patch_size]  += patch_hat
            weight[i:i+patch_size, j:j+patch_size] += 1.0

    denoised = accum / weight  # weight > 0 everywhere by construction

    active_atoms = np.sum(gamma > 1e-5)
    return np.clip(denoised, 0, 255), active_atoms


# Load test images
test_files = sorted(test_dir.glob("*.jpg"))
selected_indices = rng.choice(len(test_files), size=3, replace=False)
selected_files = [test_files[i] for i in selected_indices]

results = []

for file in selected_files:
    rgb = io.imread(file)
    clean = rgb2gray(rgb) * 255.0
    image_id = file.stem
    image_seed = int.from_bytes(image_id.encode(), "little") % (2**32)

    for sigma in sigmas:
        noise_rng = np.random.default_rng(image_seed * 1000 + sigma)
        noise = noise_rng.normal(0.0, sigma, size=clean.shape)
        noisy = np.clip(clean + noise, 0, 255)

        # Denoise using DL-SBL posterior mean
        denoised, active_atoms = denoise_image_sbl(
            noisy_img=noisy,
            D=D,
            gamma=gamma,
            sigma_n=sigma,
            patch_size=patch_size
        )

        h, w = denoised.shape
        clean_crop = clean[:h, :w]
        noisy_crop = noisy[:h, :w]

        noisy_mse = mse(clean_crop, noisy_crop)
        denoised_mse = mse(clean_crop, denoised)
        noisy_psnr = psnr(clean_crop, noisy_crop)
        denoised_psnr = psnr(clean_crop, denoised)

        results.append([
            image_id, sigma, noisy_mse, denoised_mse,
            noisy_psnr, denoised_psnr, 0.0, active_atoms
        ])

        print(
            f"{image_id}, sigma={sigma}: "
            f"Noisy PSNR={noisy_psnr:.2f}, "
            f"Denoised PSNR={denoised_psnr:.2f}, "
            f"Active Atoms={active_atoms}"
        )

# Save metrics table
results = np.array(results, dtype=object)
header = "image_id,sigma,noisy_mse,denoised_mse,noisy_psnr,denoised_psnr,epsilon,avg_atoms"
np.savetxt(output_dir / "metrics.csv", results, delimiter=",", fmt="%s", header=header, comments="")
print(f"\nSaved results to: {output_dir / 'metrics.csv'}")