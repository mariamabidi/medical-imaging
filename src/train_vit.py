import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import mlflow.pytorch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data_preprocessing import train_loader, val_loader, test_loader, class_names
from evaluate_full import evaluate_full

# ------------------------
# 1. Device Setup
# ------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# ------------------------
# 2. Model (ViT)
# ------------------------
model = timm.create_model("vit_base_patch16_224", pretrained=True)
old_head = model.head  # Save the original layer
model.head = nn.Sequential(
    nn.Dropout(0.3),
    old_head  # keep the original linear layer with correct shape
)

model = model.to(device)

# ------------------------
# 3. Loss & Optimizer
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
# 5. Training Loop with MLflow
# ------------------------
EPOCHS = 40
PATIENCE = 5
best_val_loss = np.inf
patience_counter = 0

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

mlflow.set_experiment("Brain_Tumor_ViT_Classification")

with mlflow.start_run():
    mlflow.log_param("model", "ViT-Base-16")
    mlflow.log_param("optimizer", "AdamW")
    mlflow.log_param("lr", 2e-6)
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

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)

        # ---- Early Stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_models/best_vit.pth")
            mlflow.log_artifact("best_models/best_vit.pth")
            print("✅ Model improved, saving checkpoint...")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("⏹️ Early stopping triggered!")
            break

    if os.path.exists("vit_model"):
        shutil.rmtree("vit_model")

    # Save final model
    mlflow.pytorch.save_model(model, "vit_model")
    mlflow.log_artifact("vit_model")

print("Training complete ✅")

acc_metrics = evaluate_full(
    model=model,
    test_loader=test_loader,
    class_names=class_names,
    device=device,
    model_name="vit_model"
)
