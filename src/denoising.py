import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io
from skimage.color import rgb2gray
from sklearn.linear_model import orthogonal_mp

Data_root = Path("../Data/images")
test_dir = Data_root / "test"
output_dir = Path("../Data/denoised_results")
output_dir.mkdir(exist_ok=True)

sigmas = [5, 10, 15, 25]
patch_size = 8
#epsilon=10
seed = 0
rng = np.random.default_rng(seed)

D = np.load("../Data/dictionary_D.npy") #Load dict. 64x256
print("Loaded dictionary D:", D.shape)

# helpers
def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def psnr(clean, estimate):
    error = mse(clean, estimate)
    if error == 0:
        return float("inf")
    return 10 * np.log10((255.0 ** 2) / error)

def denoise_image(noisy_img, D, patch_size=8, epsilon=10.0): #Patch extraction / reconstruction for non overlapping patches
    h, w = noisy_img.shape

    h_crop = (h // patch_size) * patch_size
    w_crop = (w // patch_size) * patch_size

    noisy_crop = noisy_img[:h_crop, :w_crop]
    denoised = np.zeros_like(noisy_crop)

    total_nonzeros = 0
    total_patches = 0

    for i in range(0, h_crop, patch_size):
        for j in range(0, w_crop, patch_size):

            patch = noisy_crop[i:i+patch_size, j:j+patch_size]

            patch_mean = np.mean(patch)
            patch_centered = patch - patch_mean

            y = patch_centered.flatten().reshape(-1, 1)
 
            x = orthogonal_mp(D, y, tol=epsilon**2) # OMP with residual threshold

            total_nonzeros += np.sum(np.abs(x) > 1e-8)
            total_patches += 1

            patch_hat = D @ x
            patch_hat = patch_hat.reshape(patch_size, patch_size)
            patch_hat = patch_hat + patch_mean

            denoised[i:i+patch_size, j:j+patch_size] = patch_hat

    avg_nonzeros = total_nonzeros / total_patches
    return np.clip(denoised, 0, 255), avg_nonzeros

# Load test images
test_files = sorted(test_dir.glob("*.jpg"))
selected_indices = rng.choice(len(test_files), size=3, replace=False) # pick 3 images
selected_files = [test_files[i] for i in selected_indices]

print("Selected test images:")
for f in selected_files:
    print(f.name)



# Denoise each image and sigma
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

        # adaptive residual threshold based on expected noise energy
        epsilon_denoise = sigma * np.sqrt(patch_size ** 2)

        denoised, avg_nonzeros = denoise_image(
            noisy_img=noisy,
            D=D,
            patch_size=patch_size,
            epsilon=epsilon_denoise
        )

        h, w = denoised.shape
        clean_crop = clean[:h, :w]
        noisy_crop = noisy[:h, :w]

        noisy_mse = mse(clean_crop, noisy_crop)
        denoised_mse = mse(clean_crop, denoised)

        noisy_psnr = psnr(clean_crop, noisy_crop)
        denoised_psnr = psnr(clean_crop, denoised)

        results.append([
            image_id,
            sigma,
            noisy_mse,
            denoised_mse,
            noisy_psnr,
            denoised_psnr,
            epsilon_denoise,
            avg_nonzeros
        ])

        print(
            f"{image_id}, sigma={sigma}: "
            f"epsilon={epsilon_denoise:.2f}, "
            f"Noisy PSNR={noisy_psnr:.2f}, "
            f"Denoised PSNR={denoised_psnr:.2f}, "
            f"Avg atoms={avg_nonzeros:.2f}"
        )


# -----------------------------
# Save results table
# -----------------------------
results = np.array(results, dtype=object)

header = "image_id,sigma,noisy_mse,denoised_mse,noisy_psnr,denoised_psnr,epsilon,avg_atoms"

np.savetxt(
    output_dir / "metrics.csv",
    results,
    delimiter=",",
    fmt="%s",
    header=header,
    comments=""
)

print("\nSaved results to:")
print(output_dir / "metrics.csv")
print(output_dir)