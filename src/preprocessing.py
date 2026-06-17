# preprocessing.py
import numpy as np
import matplotlib.pyplot as plt
import utils


#setup
rng = np.random.default_rng(42)
train_dir = utils.DATA_ROOT / "train"
test_dir = utils.DATA_ROOT / "test"

sigmas = [5, 10, 15, 25] #noise levels

# Get 3 random test images
test_files = sorted(test_dir.glob("*.jpg"))
selected_files = rng.choice(test_files, size=3, replace=False)
print("Selected test images:")
for f in selected_files:
    print(f.name)



clean_images = {file.stem: utils.load_image_gray(file) for file in selected_files}
noisy_images = {}

for image_id, clean_img in clean_images.items():
    noisy_images[image_id] = {}
    image_seed = utils.get_image_seed(image_id)

    for sigma in sigmas: # inject noise
        noise_rng = np.random.default_rng(image_seed * 1000 + sigma)
        noise = noise_rng.normal(loc= 0.0, scale = sigma, size=clean_img.shape)
        noisy_images[image_id][sigma] = np.clip(clean_img + noise, 0, 255)

# Chekc first image

example_id = list(clean_images.keys())[0]
fig, axes = plt.subplots(1, 5, figsize=(16, 4))

axes[0].imshow(clean_images[example_id], cmap="gray")
axes[0].set_title("Clean")
axes[0].axis("off")

for idx, sigma in enumerate(sigmas):
    axes[idx + 1].imshow(noisy_images[example_id][sigma], cmap="gray")
    axes[idx + 1].set_title(f"σ={sigma}")
    axes[idx + 1].axis("off")

plt.tight_layout()

plt.show()

# Training patches
patch_size = 8
num_patches = 6000

train_files = sorted(train_dir.glob("*.jpg")) #image ids
train_images = {file.stem: utils.load_image_gray(file) for file in train_files} # full file path

Y = np.zeros((patch_size**2, num_patches))#64x6000

for patch_idx in range(num_patches):
    file = rng.choice(train_files) # choose a random id
    img = train_images[file.stem]#get associated
    
    h, w = img.shape # height and width

    x = rng.integers(0, h - patch_size + 1) # left
    y = rng.integers(0, w - patch_size + 1) #top

    Y[:, patch_idx] = img[x:x+patch_size, y:y+patch_size].flatten() # choose random patch and flatten

# Plot a single  patch
idx = rng.integers(0, num_patches)
plt.imshow(Y[:, idx].reshape(patch_size, patch_size), cmap="gray")
plt.title(f"Random patch index {idx}")
plt.axis("off")
plt.show()

# Save preprocessed matrix output
np.save("../Data/Y_patches.npy", Y)
print("Saved artifacts to ../Data/Y_patches.npy")