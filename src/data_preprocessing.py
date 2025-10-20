import os
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import cv2

# ------------------------
# 1. Transformations
# ------------------------
IMG_SIZE = 224
BATCH_SIZE = 32

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),       # Random flip
    transforms.RandomRotation(15),                # Random rotation
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.2,
                           contrast=0.2,
                           saturation=0.2,
                           hue=0.1),             # Random color jitter
    transforms.Lambda(
        lambda img: torch.tensor(
            np.stack([
                         cv2.equalizeHist(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY))
                     ] * 3, axis=-1),  # replicate grayscale → 3-channel
            dtype=torch.float32
        ).permute(2, 0, 1) / 255.0
    ),    # Converts to [0,1] tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda( lambda img: torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ------------------------
# 2. Dataset from folders
# ------------------------
root_dir = "../data/brain4"

# full dataset (no transform yet, we’ll apply per split)
full_dataset = datasets.ImageFolder(root=root_dir)

# splits
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

# re-assign transforms
train_set.dataset.transform = train_transform
val_set.dataset.transform = val_test_transform
test_set.dataset.transform = val_test_transform

class_names = full_dataset.classes
print("Class mapping:", full_dataset.class_to_idx)

# ------------------------
# 3. DataLoaders
# ------------------------
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

# ------------------------
# 4. Show Sample Batch
# ------------------------
def show_batch(images, labels, classes):
    grid = torchvision.utils.make_grid(images[:16], nrow=8, normalize=True)
    npimg = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(15, 5))
    plt.imshow(npimg)
    plt.axis("off")
    plt.title("Sample Batch (First 16 Images)")
    plt.show()

    print("Labels:", [classes[int(l)] for l in labels[:16]])


# Quick Test
images, labels = next(iter(train_loader))
show_batch(images, labels, class_names)
print("Batch image tensor shape:", images.shape)
print("Batch labels shape:", labels.shape)
print("Unique labels in this batch:", labels.unique())
