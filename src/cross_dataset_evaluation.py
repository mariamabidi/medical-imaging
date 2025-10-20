import torch
import mlflow.pytorch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# --- Paths ---
test_root = "../data/combined_test_data"
models = {
    "CNN": "resnet50_model4",
    "ViT": "vit_model",
    "Swin": "swin_model"
}

# --- Data ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda( lambda img: torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root=test_root, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
classes = test_dataset.classes
print("✅ Loaded combined test data:", classes)

# --- Device ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Evaluate models ---
for model_name, path in models.items():
    print(f"\n🔍 Evaluating {model_name}...")

    model = mlflow.pytorch.load_model(path)
    model = model.to(device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # --- ROC Curve ---
    y_true_bin = label_binarize(all_labels, classes=range(len(classes)))
    all_probs = np.array(all_probs)
    plt.figure(figsize=(6, 5))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title(f"{model_name} - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()
