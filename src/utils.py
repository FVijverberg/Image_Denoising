# utils.py
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io
from skimage.color import rgb2gray

DATA_ROOT = Path("../Data/images")
OUTPUT_DIR = Path("../Data/denoised_results")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_image_gray(file_path):
    #Loads an image and converts it to grayscale scaled to 0-255.
    rgb = io.imread(file_path)
    return rgb2gray(rgb) * 255.0 #rescaled to [0,255]

def get_image_seed(image_id):
    #Get a seed based on the image name.
    return int.from_bytes(image_id.encode(), "little") % (2**32)

def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def psnr(clean, estimate):
    error = mse(clean, estimate)
    return float("inf") if error == 0 else 10 * np.log10((255.0 ** 2) / error)

def save_metrics(results, method):
    #Saves evaluation results
    df = pd.DataFrame(results, columns=[
        "image_id", "sigma", "noisy_mse", "denoised_mse",
        "noisy_psnr", "denoised_psnr", "epsilon", "avg_atoms", "denoise_time_s"
    ])
    if method == "omp":
        df.to_csv(OUTPUT_DIR / "metrics_omp.csv", index=False)
        print(f"\nSaved metrics to: {OUTPUT_DIR / 'metrics_omp.csv'}")
    else:
        df.to_csv(OUTPUT_DIR / "metrics_dlsbl.csv", index=False)
        print(f"\nSaved metrics to: {OUTPUT_DIR / 'metrics_dlsbl.csv'}")

def visualize_dictionary(D_path="../Data/dictionary_D_omp.npy", title="Trained Dictionary Atoms"):
    import matplotlib.pyplot as plt
    D = np.load(D_path)
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(D[:, i].reshape(8, 8), cmap="gray")
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()