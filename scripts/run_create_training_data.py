from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from scripts.training_formatters import TRAINING_COT_FORMATTERS, TRAINING_NO_COT_FORMATTERS
from stage_one import main

if __name__ == "__main__":
    # Script to replicate generating training data
    # Run `export PYTHONPATH=.; python scripts/run_create_training_data.py`
    exp_dir_gpt_35 = "experiments/training_data_temp_1"
    main(
        dataset="cot_training",
        formatters=[f.name() for f in TRAINING_COT_FORMATTERS + TRAINING_NO_COT_FORMATTERS],
        example_cap=5000,
        models=["gpt-3.5-turbo"],
        temperature=1.0,
        exp_dir=exp_dir_gpt_35,
        batch=10,
    )
    exp_dir_claude_2 = "experiments/training_data_temp_1_claude_2_unbiased"
    main(
        dataset="cot_training",
        formatters=[ZeroShotCOTUnbiasedFormatter.name()],
        example_cap=5000,
        models=["claude-2"],
        temperature=1.0,
        exp_dir=exp_dir_claude_2,
        batch=20,
    )

