# Medical Imaging Project

A deep learning pipeline for **brain MRI image classification**, using PyTorch. The project demonstrates data preprocessing, augmentation, training, and evaluation for medical imaging tasks.

---

## 🚀 Features

- **Data Preprocessing & Augmentation**
  - Resize, normalize, and apply on-the-fly augmentations like random flips, rotations, and color jitter to improve generalization.
  
- **Dataset Handling**
  - Automatic train/validation/test split.
  - Supports datasets organized in class-specific folders.

- **Model Training**
  - Compatible with custom CNNs and pretrained models (e.g., ResNet).
  - Efficient batch processing with PyTorch DataLoader.

- **Visualization**
  - Inspect sample batches and labels.
  - Quickly verify augmentations and data quality.
