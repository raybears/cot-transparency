from cot_transparency.formatters.core.answer_always_a import (
    AnswerAlwaysANoCOTFormatter,
    AnswerAlwaysAFormatterDifferentPrompts,
)
from cot_transparency.formatters.core.sycophancy import ZeroShotSycophancyFormatter, ZeroShotCOTSycophancyFormatter
from cot_transparency.formatters.core.unbiased import (
    ZeroShotCOTUnbiasedFormatter,
    ZeroShotUnbiasedFormatter,
)
from stage_one import main

if __name__ == "__main__":
    # Script to replicate generating training data
    # Run `export PYTHONPATH=.; python scripts/run_create_training_data.py`
    # For simple bias augmentation COT training
    exp_dir = "experiments/mmlu_answer_is_always_a"
    main(
        # dataset="cot_testing",
        # tasks=MMLU_SUPERCATEGORIES,
        tasks=["truthful_qa"],
        formatters=[
            ZeroShotCOTUnbiasedFormatter.name(),
            ZeroShotCOTSycophancyFormatter.name(),
            ZeroShotUnbiasedFormatter.name(),
            ZeroShotSycophancyFormatter.name(),
            # AnswerAlwaysAFormatter.name(),
            AnswerAlwaysAFormatterDifferentPrompts.name(),
            AnswerAlwaysANoCOTFormatter.name(),
        ],
        example_cap=400,
        models=["gpt-3.5-turbo", "ft:gpt-3.5-turbo-0613:far-ai::8GlQ0cun", "ft:gpt-3.5-turbo-0613:far-ai::8GQiNe1D"],
        temperature=0.0,
        exp_dir=exp_dir,
        batch=40,
        raise_after_retries=False,
        num_tries=1,
    )
