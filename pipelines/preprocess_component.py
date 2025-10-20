from kfp.v2.dsl import component, Output, Dataset
import subprocess

@component(
    base_image="python:3.10-slim",
    packages_to_install=["torch", "torchvision", "numpy", "pillow", "mlflow"]
)
def preprocess_op(output_data: Output[Dataset]):
    """
    Runs your data_preprocessing.py script.
    Produces preprocessed data for training.
    """
    print("📦 Running data preprocessing...")

    # Run the preprocessing script inside src/
    subprocess.run(["python", "src/data_preprocessing.py"], check=True)

    # Dummy file for Kubeflow output tracking
    with open(output_data.path, "w") as f:
        f.write("Preprocessing completed successfully.")
