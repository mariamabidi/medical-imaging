import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet50, ResNet50_Weights
import mlflow.pytorch
import numpy as np

from src.data_preprocessing import train_loader, val_loader, test_loader, class_names
from evaluate_full import evaluate_full

# ------------------------
# 1. Device
# ------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ------------------------
# 2. Model (ResNet50 pretrained on ImageNet)
# ------------------------

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Replace final layer (1000 classes → 2 classes: tumor vs no tumor)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 4))
model = model.to(device)

# ------------------------
# 3. Loss & Optimizer
# ------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000001)

scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

# ------------------------
# 4. Training & Evaluation Functions
# ------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


# ------------------------
# 5. Training Loop with MLflow
# ------------------------
EPOCHS = 25
PATIENCE = 5
best_val_loss = np.inf
patience_counter = 0

mlflow.set_experiment("Brain_Tumor_4Class_Classification")

with mlflow.start_run():
    mlflow.log_param("model", "ResNet50")
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("lr", 1e-4)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", 32)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch + 1}/{EPOCHS} "
              f"- Train loss: {train_loss:.4f}, acc: {train_acc:.4f} "
              f"- Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)

        # ---- Early Stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_models/best_resnet.pth")
            mlflow.log_artifact("best_models/best_resnet.pth")
            print("✅ Model improved, saving checkpoint...")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("⏹️ Early stopping triggered!")
            break

    if os.path.exists("resnet50_model4"):
        shutil.rmtree("resnet50_model4")

    # Save to local folder
    mlflow.pytorch.save_model(model, "resnet50_model4")

    # Log it as an artifact
    mlflow.log_artifact("resnet50_model4")

acc_metrics = evaluate_full(
    model=model,
    test_loader=test_loader,
    class_names=class_names,
    device=device,
    model_name="resnet50_model4"
)
