# plot_results.py — all visualisations for the image denoising project.
#
# Sections run in order:
#   1. Metrics comparison  (PSNR, MSE, PSNR gain, sparsity) from CSV files
#   2. Visual comparison   (clean / noisy / OMP / SBL denoised images)
#   3. Dictionary atoms    (K-SVD and DL-SBL side-by-side)
#   4. EM convergence      (needs a training log; pass path as CLI arg)
#
# Usage:
#   python plot_results.py                    # sections 1-3
#   python plot_results.py train_log.txt      # all four sections

import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
from denoising import denoise_image

PATCH = 8
SIGMA_VIS = 25     # noise level used for visual comparison
STRIDE_VIS = 4
EM_TOL = 1e-3      # must match em_tol used during training
COLORS = {5: "#1f77b4", 10: "#ff7f0e", 15: "#2ca02c", 25: "#d62728"}

# -----------------------------------------------------------------------
# Section 1: Metrics comparison
# -----------------------------------------------------------------------

omp_df = pd.read_csv(utils.OUTPUT_DIR / "metrics_omp.csv")
sbl_df = pd.read_csv(utils.OUTPUT_DIR / "metrics_dlsbl.csv")

omp = omp_df.groupby("sigma").mean(numeric_only=True).reset_index()
sbl = sbl_df.groupby("sigma").mean(numeric_only=True).reset_index()

print("\nOMP Summary")
print(omp)
print("\nDL-SBL Summary")
print(sbl)

# PSNR vs sigma
plt.figure(figsize=(6, 4))
plt.plot(omp["sigma"], omp["noisy_psnr"], "k--o", linewidth=2, label="Noisy")
plt.plot(omp["sigma"], omp["denoised_psnr"], "o-", linewidth=2, label="K-SVD/OMP")
plt.plot(sbl["sigma"], sbl["denoised_psnr"], "s-", linewidth=2, label="DL-SBL")
plt.xlabel("Noise level σ")
plt.ylabel("PSNR (dB)")
plt.title("PSNR vs Noise Level")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(utils.OUTPUT_DIR / "psnr_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# MSE vs sigma
plt.figure(figsize=(6, 4))
plt.plot(omp["sigma"], omp["noisy_mse"], "k--o", linewidth=2, label="Noisy")
plt.plot(omp["sigma"], omp["denoised_mse"], "o-", linewidth=2, label="K-SVD/OMP")
plt.plot(sbl["sigma"], sbl["denoised_mse"], "s-", linewidth=2, label="DL-SBL")
plt.xlabel("Noise level σ")
plt.ylabel("MSE")
plt.title("MSE vs Noise Level")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(utils.OUTPUT_DIR / "mse_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# PSNR gain bar chart
omp_gain = omp["denoised_psnr"] - omp["noisy_psnr"]
sbl_gain = sbl["denoised_psnr"] - sbl["noisy_psnr"]
x = np.arange(len(omp["sigma"]))
width = 0.35

plt.figure(figsize=(6, 4))
plt.bar(x - width / 2, omp_gain, width, label="K-SVD/OMP")
plt.bar(x + width / 2, sbl_gain, width, label="DL-SBL")
plt.xticks(x, omp["sigma"])
plt.xlabel("Noise level σ")
plt.ylabel("PSNR gain (dB)")
plt.title("PSNR Improvement")
plt.grid(axis="y")
plt.legend()
plt.tight_layout()
plt.savefig(utils.OUTPUT_DIR / "psnr_gain_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# Sparsity comparison
plt.figure(figsize=(6, 4))
plt.plot(omp["sigma"], omp["avg_atoms"], "o-", linewidth=2, label="K-SVD/OMP")
plt.plot(sbl["sigma"], sbl["avg_atoms"], "s-", linewidth=2, label="DL-SBL")
plt.xlabel("Noise level σ")
plt.ylabel("Average active atoms")
plt.title("Sparsity Comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(utils.OUTPUT_DIR / "sparsity_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved metrics plots.")

# -----------------------------------------------------------------------
# Section 2: Visual comparison (same image and noise seed as denoising.py)
# -----------------------------------------------------------------------

test_files = sorted((utils.DATA_ROOT / "test").glob("*.jpg"))
rng = np.random.default_rng(42)
selected_files = rng.choice(test_files, size=3, replace=False)
file = selected_files[1]

clean = utils.load_image_gray(file)
img_seed = utils.get_image_seed(file.stem)
noise_rng = np.random.default_rng(img_seed * 1000 + SIGMA_VIS)
noisy = np.clip(clean + noise_rng.normal(0, SIGMA_VIS, clean.shape), 0, 255)

D_omp = np.load("../Data/dictionary_D_omp.npy")
D_sbl = np.load(f"../Data/dictionary_D_sbl_{SIGMA_VIS}.npy")

denoised_omp, _ = denoise_image(noisy, D_omp, "omp", sigma=SIGMA_VIS, stride=STRIDE_VIS)
denoised_sbl, _ = denoise_image(noisy, D_sbl, "sbl", sigma=SIGMA_VIS, stride=STRIDE_VIS)

psnr_noisy = utils.psnr(clean, noisy)
psnr_omp = utils.psnr(clean, denoised_omp)
psnr_sbl = utils.psnr(clean, denoised_sbl)

images = [clean, noisy, denoised_omp, denoised_sbl]
labels = ["clean", f"noisy_sigma{SIGMA_VIS}", f"omp_sigma{SIGMA_VIS}", f"sbl_sigma{SIGMA_VIS}"]

for img, label in zip(images, labels):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    plt.tight_layout(pad=0)
    out = utils.OUTPUT_DIR / f"{file.stem}_{label}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

print(f"Visual PSNR: noisy {psnr_noisy:.2f} | OMP {psnr_omp:.2f} | SBL {psnr_sbl:.2f} dB")

# -----------------------------------------------------------------------
# Section 3: Dictionary atoms
# -----------------------------------------------------------------------

def plot_atom_sample(dict_path, title, out_name, rows=4, cols=8):
    D = np.load(dict_path)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.85, rows * 0.85))
    for i, ax in enumerate(axes.flat):
        atom = D[:, i].reshape(PATCH, PATCH)
        atom = (atom - atom.min()) / (np.ptp(atom) + 1e-12)
        ax.imshow(atom, cmap="gray")
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.92,
                        wspace=0.08, hspace=0.08)
    fig.savefig(utils.OUTPUT_DIR / out_name, dpi=150)
    plt.close(fig)
    print(f"Saved {utils.OUTPUT_DIR / out_name}")

plot_atom_sample("../Data/dictionary_D_omp.npy", "K-SVD", "dict_ksvd.png")
plot_atom_sample(f"../Data/dictionary_D_sbl_5.npy", r"DL-SBL ($\sigma=5$)", "dict_sbl_5.png")

# -----------------------------------------------------------------------
# Section 4: EM convergence (only if a log file is provided)
# -----------------------------------------------------------------------

HEADER = re.compile(r"sigma=(\d+)")
LINE = re.compile(r"EM\s+(\d+)/\d+\s*\|\s*recon MSE\s+([\d.]+)\s*\|\s*rel_change\s+([\d.eE+-]+)")


def parse_log(path):
    history = {}
    current = None
    with open(path) as f:
        for line in f:
            header = HEADER.search(line)
            if header:
                current = int(header.group(1))
                history[current] = ([], [], [])
                continue
            m = LINE.search(line)
            if m and current is not None:
                it, mse, rel = int(m.group(1)), float(m.group(2)), float(m.group(3))
                history[current][0].append(it)
                history[current][1].append(mse)
                history[current][2].append(rel)
    return history


def plot_convergence_curve(history, index, ylabel, title, out_name, tol_line=False):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for sigma in sorted(history):
        iters, values = history[sigma][0], history[sigma][index]
        ax.plot(iters, values, color=COLORS.get(sigma), label=fr"$\sigma={sigma}$")
    ax.set_yscale("log")
    if tol_line:
        ax.axhline(EM_TOL, ls="--", color="gray", lw=1)
        ax.text(150, EM_TOL * 1.13, r"convergence tol $10^{-3}$", color="gray", fontsize=9)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(utils.OUTPUT_DIR / out_name, dpi=150)
    plt.close(fig)
    print(f"Saved {utils.OUTPUT_DIR / out_name}")


log_path = sys.argv[1] if len(sys.argv) > 1 else None
if log_path:
    history = parse_log(log_path)
    plot_convergence_curve(history, 1, "Reconstruction MSE", "Training reconstruction MSE",
                           "convergence_mse.png")
    plot_convergence_curve(history, 2, "Relative change", "Convergence criterion",
                           "convergence_rel.png", tol_line=True)
else:
    print("No training log provided — skipping convergence plots. Pass log path as argument.")
