import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------
# 1. Load MLflow Experiment Data
# ------------------------
experiments = [
    "Brain_Tumor_4Class_Classification",  # CNN
    "Brain_Tumor_ViT_Classification",     # ViT
    "Brain_Tumor_Swin"                    # Swin Transformer
]

results = []

for exp_name in experiments:
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp:
        runs_df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        # get best val_acc
        best_run = runs_df.sort_values("metrics.val_acc", ascending=False).iloc[0]
        results.append({
            "Model": exp_name.replace("Brain_Tumor_", "").replace("_Classification", ""),
            "Val Accuracy": best_run["metrics.val_acc"],
            "Train Accuracy": best_run["metrics.train_acc"],
            "Val Loss": best_run["metrics.val_loss"],
            "Train Loss": best_run["metrics.train_loss"]
        })
    else:
        print(f"⚠️ Experiment '{exp_name}' not found in MLflow.")

# Convert to DataFrame
df = pd.DataFrame(results)
print("\n=== Model Comparison (From MLflow) ===")
print(df)

# ------------------------
# 2. Scatter Plot Comparison
# ------------------------
plt.figure(figsize=(8, 6))

# Validation scatter
plt.scatter(df["Val Loss"], df["Val Accuracy"], color='blue', label="Validation", s=100, marker='o')

# Training scatter
plt.scatter(df["Train Loss"], df["Train Accuracy"], color='orange', label="Training", s=100, marker='s')

# Annotate model names
for i in range(len(df)):
    plt.text(df["Val Loss"][i] + 0.002, df["Val Accuracy"][i] + 0.002, df["Model"][i], fontsize=10, color='blue')
    plt.text(df["Train Loss"][i] + 0.002, df["Train Accuracy"][i] - 0.01, df["Model"][i], fontsize=10, color='orange')

# Labels and styling
plt.title("Model Performance Comparison (Accuracy vs Loss)", fontsize=14)
plt.xlabel("Loss")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
