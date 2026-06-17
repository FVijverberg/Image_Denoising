# Image Denoising via Dictionary Learning

Compares **K-SVD + OMP** and **DL-SBL** for patch-based image denoising on BSDS images.
Patches are 8×8 pixels; dictionaries have 128 atoms; noise levels σ ∈ {5, 10, 15, 25}.

## Setup

```bash
pip install -r requirements.txt
```

Data goes in `../Data/images/{train,test}/*.jpg` (one level above this folder).
All scripts are run from inside `src/`.

```bash
cd src
```

---

## Run order

### 1. Extract training patches

```bash
python preprocessing.py
```

Picks 3 test images, creates noisy versions at each σ, extracts 6000 random 8×8 patches → `../Data/Y_patches.npy`.

---

### 2a. Train K-SVD dictionary

```bash
python dictionary_learning.py
```

Runs 20 iterations of K-SVD with OMP sparse coding. Takes ~2 min.
Saves `../Data/dictionary_D_omp.npy`.

---

### 2b. Train DL-SBL dictionaries

```bash
python dict_learning_dlsbl.py
```

Trains a separate dictionary for each σ (up to 500 EM iterations each). Takes ~40 min per σ.
Saves `../Data/dictionary_D_sbl_{5,10,15,25}.npy`.

To save the convergence log for plotting later:

```bash
python -u dict_learning_dlsbl.py > train_log.txt
```

---

### 3. Denoise test images

Edit the `method` variable at the top of `denoising.py` (`"omp"` or `"sbl"`), then:

```bash
python denoising.py
```

Saves PSNR/MSE results to `../Data/denoised_results/metrics_{omp,dlsbl}.csv`.
Run once per method.

---

### 4. Plot results

```bash
python plot_results.py                   # PSNR/MSE curves, visual comparison, dictionary atoms
python plot_results.py train_log.txt     # also EM convergence curves (needs the log from step 2b)
```

Saves all figures to `../Data/denoised_results/`.

---

## Output files

| File | Description |
|------|-------------|
| `metrics_omp.csv` / `metrics_dlsbl.csv` | PSNR, MSE, sparsity per image and σ |
| `psnr_comparison.png` | PSNR vs σ for both methods |
| `mse_comparison.png` | MSE vs σ |
| `psnr_gain_comparison.png` | PSNR improvement bar chart |
| `sparsity_comparison.png` | Average active atoms per patch |
| `{image}_{clean,noisy,omp,sbl}_sigma25.png` | Visual comparison at σ=25 |
| `dict_ksvd.png` / `dict_sbl_5.png` | Learned dictionary atoms |
| `convergence_mse.png` / `convergence_rel.png` | EM training curves (if log provided) |

---