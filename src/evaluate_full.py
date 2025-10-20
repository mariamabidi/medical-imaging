import os
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import mlflow


def evaluate_full(model, test_loader, class_names, device, model_name):
    """
    Evaluate a trained model on the test set.
    Logs confusion matrix, ROC curve, metrics, and classification report to MLflow.
    Appends to classification_report.txt across runs.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    # ------------------------
    # 1️⃣ Forward Pass - Gather predictions
    # ------------------------
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # ------------------------
    # 2️⃣ Metrics
    # ------------------------
    acc = np.mean(all_preds == all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4, output_dict=True
    )

    # Log overall + per-class metrics
    mlflow.log_metric("test_acc", acc)
    for cls_name in class_names:
        mlflow.log_metric(f"{cls_name}_precision", report[cls_name]["precision"])
        mlflow.log_metric(f"{cls_name}_recall", report[cls_name]["recall"])
        mlflow.log_metric(f"{cls_name}_f1", report[cls_name]["f1-score"])

    # ------------------------
    # 3️⃣ Confusion Matrix
    # ------------------------
    os.makedirs("images", exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    cm_path = f"images/confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # ------------------------
    # 4️⃣ ROC Curve (multi-class)
    # ------------------------
    y_true_bin = label_binarize(all_labels, classes=np.arange(len(class_names)))
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro-average AUC
    macro_auc = np.mean(list(roc_auc.values()))
    mlflow.log_metric("macro_roc_auc", macro_auc)

    plt.figure(figsize=(7, 6))
    for i, cls in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], label=f"{cls} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.title(f"ROC Curves - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    roc_path = f"images/roc_curve_{model_name.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    mlflow.log_artifact(roc_path)

    # ------------------------
    # 5️⃣ Classification Report (append mode)
    # ------------------------
    REPORT_DIR = "../results"
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "classification_report.txt")

    with open(report_path, "a") as f:
        f.write("\n\n==============================\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write("==============================\n")
        for cls, metrics in report.items():
            f.write(f"{cls}: {metrics}\n")
    mlflow.log_artifact(report_path)

    # ------------------------
    # 6️⃣ Print Summary
    # ------------------------
    print(f"\n✅ Test Accuracy: {acc:.4f}")
    print(f"Macro AUC: {macro_auc:.4f}")
    print("\nClassification Report:\n",
          classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    return {
        "accuracy": acc,
        "macro_auc": macro_auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }
