# plot_results.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils

omp_df = pd.read_csv(utils.OUTPUT_DIR / "metrics_omp.csv")
sbl_df = pd.read_csv(utils.OUTPUT_DIR / "metrics_dlsbl.csv")

omp = omp_df.groupby("sigma").mean(numeric_only=True).reset_index()
sbl = sbl_df.groupby("sigma").mean(numeric_only=True).reset_index()

print("\nOMP Summary")
print(omp)

print("\nDL-SBL Summary")
print(sbl)

plt.figure(figsize=(6,4))

plt.plot(
    omp["sigma"],
    omp["noisy_psnr"],
    "k--o",
    linewidth=2,
    label="Noisy"
)

plt.plot(
    omp["sigma"],
    omp["denoised_psnr"],
    "o-",
    linewidth=2,
    label="K-SVD/OMP"
)

plt.plot(
    sbl["sigma"],
    sbl["denoised_psnr"],
    "s-",
    linewidth=2,
    label="DL-SBL"
)

plt.xlabel("Noise level σ")
plt.ylabel("PSNR (dB)")
plt.title("PSNR vs Noise Level")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(
    utils.OUTPUT_DIR / "psnr_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.figure(figsize=(6,4))

plt.plot(
    omp["sigma"],
    omp["noisy_mse"],
    "k--o",
    linewidth=2,
    label="Noisy"
)

plt.plot(
    omp["sigma"],
    omp["denoised_mse"],
    "o-",
    linewidth=2,
    label="K-SVD/OMP"
)

plt.plot(
    sbl["sigma"],
    sbl["denoised_mse"],
    "s-",
    linewidth=2,
    label="DL-SBL"
)

plt.xlabel("Noise level σ")
plt.ylabel("MSE")
plt.title("MSE vs Noise Level")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(
    utils.OUTPUT_DIR / "mse_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

omp_gain = omp["denoised_psnr"] - omp["noisy_psnr"]
sbl_gain = sbl["denoised_psnr"] - sbl["noisy_psnr"]

x = np.arange(len(omp["sigma"]))
width = 0.35

plt.figure(figsize=(6,4))

plt.bar(
    x - width/2,
    omp_gain,
    width,
    label="K-SVD/OMP"
)

plt.bar(
    x + width/2,
    sbl_gain,
    width,
    label="DL-SBL"
)

plt.xticks(x, omp["sigma"])

plt.xlabel("Noise level σ")
plt.ylabel("PSNR gain (dB)")
plt.title("PSNR Improvement")
plt.grid(axis="y")
plt.legend()

plt.tight_layout()
plt.savefig(
    utils.OUTPUT_DIR / "psnr_gain_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

# --------------------------------------------------
# Figure 4: Sparsity Comparison
# --------------------------------------------------

plt.figure(figsize=(6,4))

plt.plot(
    omp["sigma"],
    omp["avg_atoms"],
    "o-",
    linewidth=2,
    label="K-SVD/OMP"
)

plt.plot(
    sbl["sigma"],
    sbl["avg_atoms"],
    "s-",
    linewidth=2,
    label="DL-SBL"
)

plt.xlabel("Noise level σ")
plt.ylabel("Average active atoms")
plt.title("Sparsity Comparison")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(
    utils.OUTPUT_DIR / "sparsity_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()