import numpy as np
from pathlib import Path
from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

seed = 0 # to reproduce
rng = np.random.default_rng(seed)

dataRoot = Path("../Data/images")
trainDir = dataRoot / "train"
testDir = dataRoot / "test"

sigmas = [5, 10, 15, 25] # Noise levels

test_files = sorted(testDir.glob("*.jpg")) # load test images
print(f"Found {len(test_files)} test images")

selected_indices = rng.choice(len(test_files), size=3, replace=False) # pick 3 images

selected_files = [test_files[i] for i in selected_indices]

print("Selected test images:")
for f in selected_files:
    print(f.name)

clean_images = {} # Load clean images

for file in selected_files:
    rgb = io.imread(file)
    grey = rgb2gray(rgb)*255.0 #func returns between 0 and 1, so gotta scale back

    clean_images[file.stem] = grey

#add  noise
noisy_images = {}
for image_id, clean_img in clean_images.items():

    noisy_images[image_id] = {} # init

    image_seed =int.from_bytes(image_id.encode(), "little") % (2**32)     # fixed per-image seed

    for sigma in sigmas:
        noise_rng = np.random.default_rng(image_seed*1000 + sigma)
        noise = noise_rng.normal(loc=0.0,scale=sigma,size=clean_img.shape)
        noisy = np.clip(clean_img + noise,0, 255) # z = clip(x + N(0, sigma^2), 0, 255)
        noisy_images[image_id][sigma] = noisy

#check
example_id = list(clean_images.keys())[0]
fig, axes = plt.subplots(1, 5, figsize=(16, 4))
axes[0].imshow(clean_images[example_id], cmap="gray")
axes[0].set_title("Clean")
axes[0].axis("off")

for idx, sigma in enumerate(sigmas):
    axes[idx + 1].imshow(
        noisy_images[example_id][sigma],
        cmap="gray"
    )
    axes[idx + 1].set_title(f"σ={sigma}")
    axes[idx + 1].axis("off")

plt.tight_layout()
plt.show()
#-----------------------------------------------------------
# Extract patches from train set
patch_size = 8
num_patches = 6000

train_files = sorted(trainDir.glob("*.jpg"))
print(f"Found {len(train_files)} train images")

# init
Y = np.zeros((patch_size**2, num_patches))
train_images = {}

# convert to greyscale
for file in train_files:
    rgb = io.imread(file)
    grey = rgb2gray(rgb) * 255.0  # keep 0–255 scale

    train_images[file.stem] = grey

# extract random patches
patch_idx = 0

while patch_idx < num_patches: # iterates until at 6000

    # pick random image
    file = rng.choice(train_files)
    img = train_images[file.stem]

    h, w = img.shape

    # random crop location
    x = rng.integers(0, h - patch_size+1)
    y = rng.integers(0, w - patch_size+1)

    patch = img[x:x+patch_size, y:y+patch_size]

    Y[:, patch_idx] = patch.flatten() # vectorize

    patch_idx += 1

# check
print("\n--- Sanity Check ---")

print(f"Y shape: {Y.shape}")
print(f"Min value: {Y.min():.2f}")
print(f"Max value: {Y.max():.2f}")
zero_patches = np.sum(np.linalg.norm(Y, axis=0) == 0)
print(f"Zero patches: {zero_patches}")

idx = rng.integers(0, num_patches)
patch = Y[:, idx].reshape(patch_size, patch_size)

plt.imshow(patch, cmap="gray")
plt.title(f"Random patch index {idx}")
plt.axis("off")
plt.show()

np.save("../Data/Y_patches.npy", Y)