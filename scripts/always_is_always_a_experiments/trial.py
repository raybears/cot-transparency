from cot_transparency.formatters.core.answer_always_a import AnswerAlwaysAFormatter, AnswerAlwaysANoCOTFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter, ZeroShotUnbiasedFormatter
from stage_one import main

if __name__ == "__main__":
    # Script to replicate generating training data
    # Run `export PYTHONPATH=.; python scripts/run_create_training_data.py`
    # For simple bias augmentation COT training
    exp_dir = "experiments/trial_answer_a"
    main(
        dataset="cot_testing",
        formatters=[
            ZeroShotCOTUnbiasedFormatter.name(),
            # ZeroShotCOTSycophancyFormatter.name(),
            ZeroShotUnbiasedFormatter.name(),
            AnswerAlwaysAFormatter.name(),
            AnswerAlwaysANoCOTFormatter.name(),
        ],
        example_cap=400,
        models=["gpt-3.5-turbo"],
        temperature=1.0,
        exp_dir=exp_dir,
        batch=1,
    )
