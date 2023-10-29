from cot_transparency.formatters.core.unbiased import (
    ZeroShotUnbiasedFormatter,
    ZeroShotCOTUnbiasedFormatter,
)
from stage_one import main

if __name__ == "__main__":
    # Script to replicate generating training data
    exp_dir_gpt_35 = "experiments/training_data_1_unfiltered"
    main(
        dataset="cot_training",
        formatters=[ZeroShotCOTUnbiasedFormatter.name(), ZeroShotUnbiasedFormatter.name()],
        example_cap=5000,
        models=["gpt-3.5-turbo"],
        temperature=1.0,
        exp_dir=exp_dir_gpt_35,
        batch=40,
        raise_after_retries=False,
        num_retries=0,
        # High max tokens so that it does not get truncated
        max_tokens=2000,
    )
