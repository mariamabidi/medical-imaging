import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm
import mlflow.pytorch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data_preprocessing import train_loader, val_loader, test_loader, class_names
from evaluate_full import evaluate_full

# ------------------------
# 1. Device
# ------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# ------------------------
# 2. Model (Swin Transformer)
# ------------------------
model_name = "swin_tiny_patch4_window7_224"
model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))

old_head = model.head  # Save the original layer
model.head = nn.Sequential(
    nn.Dropout(0.3),
    old_head  # keep the original linear layer with correct shape
)

model = model.to(device)

x = torch.randn(4, 3, 224, 224).to(device)
out = model(x)
print("Output shape:", out.shape)

# ------------------------
# 3. Loss, Optimizer, Scheduler
# ------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-6, weight_decay=1e-4)

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
# 5. Training Loop with Early Stopping & MLflow
# ------------------------
EPOCHS = 40
PATIENCE = 7
best_val_loss = np.inf
patience_counter = 0

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

mlflow.set_experiment("Brain_Tumor_Swin")

with mlflow.start_run():
    mlflow.log_param("model", model_name)
    mlflow.log_param("optimizer", "AdamW")
    mlflow.log_param("lr", 2e-6)
    mlflow.log_param("weight_decay", 1e-4)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", train_loader.batch_size)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        # Step LR scheduler
        scheduler.step()

        print(f"Epoch {epoch + 1}/{EPOCHS} "
              f"- Train loss: {train_loss:.4f}, acc: {train_acc:.4f} "
              f"- Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # Log to MLflow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)

        # ---- Early Stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_models/best_swin.pth")
            mlflow.log_artifact("best_models/best_swin.pth")
            print("✅ Model improved, saving checkpoint...")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("⏹️ Early stopping triggered!")
            break

    if os.path.exists("swin_model"):
        shutil.rmtree("swin_model")

    # Save final model
    mlflow.pytorch.save_model(model, "swin_model")
    mlflow.log_artifact("swin_model")

acc_metrics = evaluate_full(
    model=model,
    test_loader=test_loader,
    class_names=class_names,
    device=device,
    model_name="swin_model"
)
