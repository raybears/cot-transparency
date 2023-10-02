from scripts.training_formatters import TRAINING_COT_FORMATTERS, TRAINING_DECEPTIVE_COT
from stage_one import main

if __name__ == "__main__":
    # Script to replicate generating training data
    # Run `export PYTHONPATH=.; python scripts/run_create_training_data.py`
    models = [
        "gpt-3.5-turbo",
    ]
    exp_dir = "training_data_temp_1"
    main(
        dataset="cot_training",
        formatters=[TRAINING_DECEPTIVE_COT.name()],
        example_cap=5000,
        models=models,
        temperature=1.0,
        exp_dir="experiments/deceptive_data_temp_1",
    )
