import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Load metrics
# -----------------------------
df = pd.read_csv("../Data/denoised_results/metrics.csv")

# Noise variance
df["variance"] = df["sigma"]**2

# -----------------------------
# Average over the 3 images
# -----------------------------
summary = df.groupby("variance").agg({
    "noisy_psnr":"mean",
    "denoised_psnr":"mean",
    "noisy_mse":"mean",
    "denoised_mse":"mean",
    "avg_atoms":"mean"
}).reset_index()

print(summary)

# -----------------------------
# Plot 1: PSNR vs Noise Variance
# -----------------------------
plt.figure(figsize=(7,5))

plt.plot(
    summary["variance"],
    summary["noisy_psnr"],
    marker='o',
    linewidth=2,
    label="Noisy"
)

plt.plot(
    summary["variance"],
    summary["denoised_psnr"],
    marker='s',
    linewidth=2,
    label="Denoised"
)

plt.xlabel("Noise Variance ($\\sigma^2$)")
plt.ylabel("PSNR (dB)")
plt.title("PSNR vs Noise Variance")
plt.grid(True)
plt.legend()

plt.savefig(
    "../Data/denoised_results/PSNR_vs_variance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------
# Plot 2: MSE vs Noise Variance
# -----------------------------
plt.figure(figsize=(7,5))

plt.plot(
    summary["variance"],
    summary["noisy_mse"],
    marker='o',
    linewidth=2,
    label="Noisy"
)

plt.plot(
    summary["variance"],
    summary["denoised_mse"],
    marker='s',
    linewidth=2,
    label="Denoised"
)

plt.xlabel("Noise Variance ($\\sigma^2$)")
plt.ylabel("MSE")
plt.title("MSE vs Noise Variance")
plt.grid(True)
plt.legend()

plt.savefig(
    "../Data/denoised_results/MSE_vs_variance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------
# Plot 3: Average atoms used
# -----------------------------
plt.figure(figsize=(7,5))

plt.plot(
    summary["variance"],
    summary["avg_atoms"],
    marker='o',
    linewidth=2,
    color='darkred'
)

plt.xlabel("Noise Variance ($\\sigma^2$)")
plt.ylabel("Average Atoms Selected")
plt.title("Average OMP Atoms vs Noise Variance")
plt.grid(True)

plt.savefig(
    "../Data/denoised_results/Atoms_vs_variance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------
# Plot 4: PSNR Improvement
# -----------------------------
summary["improvement"] = (
    summary["denoised_psnr"]
    - summary["noisy_psnr"]
)

plt.figure(figsize=(7,5))

plt.bar(
    summary["variance"].astype(str),
    summary["improvement"],
    color="steelblue"
)

plt.xlabel("Noise Variance ($\\sigma^2$)")
plt.ylabel("PSNR Improvement (dB)")
plt.title("PSNR Gain after Denoising")

plt.grid(axis='y')

plt.savefig(
    "../Data/denoised_results/PSNR_improvement.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nPlots saved successfully!")