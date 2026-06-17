import numpy as np
import matplotlib.pyplot as plt
import utils
from denoising import denoise_image

SIGMA = 25
STRIDE = 4

# Same selection logic as denoising.py
test_files = sorted((utils.DATA_ROOT / "test").glob("*.jpg"))
rng = np.random.default_rng(42)
selected_files = rng.choice(test_files, size=3, replace=False)
file = selected_files[1]

clean = utils.load_image_gray(file)
img_seed = utils.get_image_seed(file.stem)
noise_rng = np.random.default_rng(img_seed * 1000 + SIGMA)
noisy = np.clip(clean + noise_rng.normal(0, SIGMA, clean.shape), 0, 255)

D_omp = np.load("../Data/dictionary_D_omp.npy")
D_sbl = np.load(f"../Data/dictionary_D_sbl_{SIGMA}.npy")

denoised_omp, _ = denoise_image(noisy, D_omp, "omp", sigma=SIGMA, stride=STRIDE)
denoised_sbl, _ = denoise_image(noisy, D_sbl, "sbl", sigma=SIGMA, stride=STRIDE)

psnr_noisy = utils.psnr(clean, noisy)
psnr_omp   = utils.psnr(clean, denoised_omp)
psnr_sbl   = utils.psnr(clean, denoised_sbl)

images = [clean, noisy, denoised_omp, denoised_sbl]
titles = [
    "Clean",
    f"Noisy (σ={SIGMA})\nPSNR {psnr_noisy:.2f} dB",
    f"K-SVD/OMP\nPSNR {psnr_omp:.2f} dB",
    f"DL-SBL\nPSNR {psnr_sbl:.2f} dB",
]

labels = ["clean", f"noisy_sigma{SIGMA}", f"omp_sigma{SIGMA}", f"sbl_sigma{SIGMA}"]

for img, label in zip(images, labels):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    plt.tight_layout(pad=0)
    out = utils.OUTPUT_DIR / f"{file.stem}_{label}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")