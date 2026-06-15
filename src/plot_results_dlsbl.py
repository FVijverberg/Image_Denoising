import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

results_file = Path("../Data/denoised_results/metrics.csv")

df = pd.read_csv(results_file)

# --------------------------------------------------
# Average PSNR over images
# --------------------------------------------------

summary = (
    df.groupby("sigma")
      .agg({
          "noisy_psnr": "mean",
          "denoised_psnr": "mean"
      })
      .reset_index()
)

plt.figure(figsize=(6,4))
plt.plot(summary["sigma"],
         summary["noisy_psnr"],
         marker="o",
         label="Noisy")

plt.plot(summary["sigma"],
         summary["denoised_psnr"],
         marker="o",
         label="DL-SBL")

plt.xlabel("Noise sigma")
plt.ylabel("Average PSNR (dB)")
plt.title("Denoising Performance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# PSNR improvement
# --------------------------------------------------

summary["gain"] = (
    summary["denoised_psnr"]
    - summary["noisy_psnr"]
)

plt.figure(figsize=(6,4))
plt.plot(summary["sigma"],
         summary["gain"],
         marker="o")

plt.xlabel("Noise sigma")
plt.ylabel("PSNR Gain (dB)")
plt.title("DL-SBL Improvement")
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Per-image results
# --------------------------------------------------

for image_id in df["image_id"].unique():

    subset = df[df["image_id"] == image_id]

    plt.figure(figsize=(6,4))

    plt.plot(
        subset["sigma"],
        subset["noisy_psnr"],
        marker="o",
        label="Noisy"
    )

    plt.plot(
        subset["sigma"],
        subset["denoised_psnr"],
        marker="o",
        label="DL-SBL"
    )

    plt.title(image_id)
    plt.xlabel("Noise sigma")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Console summary
# --------------------------------------------------

print("\nAverage Results")
print(summary)

print(
    "\nMean PSNR Gain:",
    summary["gain"].mean()
)