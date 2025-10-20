from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics
import subprocess

@component(
    base_image="python:3.10-slim",
    packages_to_install=["torch", "torchvision", "timm", "mlflow", "numpy"]
)
def train_op(
    input_data: Input[Dataset],
    output_model: Output[Model],
    output_metrics: Output[Metrics],
    model_name: str = "vit"
):
    """
    Trains the specified model (ResNet, ViT, Swin, etc.)
    by calling its respective training script inside src/.
    """
    print(f"🚀 Training {model_name} model...")

    # Dynamically call your training file (e.g., src/train_vit.py or src/train_swin.py)
    subprocess.run(["python", f"src/train_{model_name}.py"], check=True)

    # Log dummy model output
    with open(output_model.path, "w") as f:
        f.write(f"{model_name} model trained successfully.")

    # Log dummy metric
    output_metrics.log_metric("train_complete", 1.0)
