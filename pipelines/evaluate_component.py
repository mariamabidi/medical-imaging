from kfp.dsl import Output
from kfp.v2.dsl import component, Input, Model, Metrics
import subprocess

@component(
    base_image="python:3.10-slim",
    packages_to_install=["torch", "torchvision", "scikit-learn", "mlflow", "matplotlib", "seaborn", "numpy"]
)
def evaluate_op(
    input_model: Input[Model],
    output_metrics: Output[Metrics]
):
    """
    Evaluates the trained model using your evaluate_full.py script.
    """
    print("📊 Running evaluation...")
    subprocess.run(["python", "src/evaluate_full.py"], check=True)

    # Dummy metric for Kubeflow
    output_metrics.log_metric("evaluation_complete", 1.0)
