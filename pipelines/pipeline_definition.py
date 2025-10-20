from kfp import dsl
from preprocess_component import preprocess_op
from train_component import train_op
from evaluate_component import evaluate_op

@dsl.pipeline(
    name="brain-tumor-classification-pipeline",
    description="End-to-end pipeline for brain tumor MRI classification"
)
def brain_tumor_pipeline(model_name: str = "vit"):
    # Step 1: Preprocessing
    preprocess_task = preprocess_op()

    # Step 2: Training (depends on preprocessing)
    train_task = train_op(
        input_data=preprocess_task.outputs["output_data"],
        model_name=model_name
    )

    # Step 3: Evaluation (depends on training)
    evaluate_task = evaluate_op(
        input_model=train_task.outputs["output_model"]
    )
