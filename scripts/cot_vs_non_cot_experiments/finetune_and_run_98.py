from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    RandomSampler,
    fine_tune_with_bias_augmentation,
)
from scripts.training_formatters import HAS_STRONG_EFFECT_FEW_SHOT_FORMATTERS
from stage_one import main

if __name__ == "__main__":
    COT_PERCENTAGE = 0.98
    for n in [100, 1000, 10000]:
        model = fine_tune_with_bias_augmentation(
            model="gpt-3.5-turbo",
            n_epochs=1,
            n_samples=n,
            post_hoc=False,
            # train on zero shot, test on few shot
            sampler=RandomSampler(formatter_options=FormatterOptions.zero_shot),
            cot_percentage=COT_PERCENTAGE,
            data_from_options=DataFromOptions.gpt_35_turbo,
            ask_to_validate_training=False,
        )
        # test on TRAINING_COT_FORMATTERS_FEW_SHOT
        test_formatters = [f.name() for f in HAS_STRONG_EFFECT_FEW_SHOT_FORMATTERS]
        main(
            exp_dir="experiments/finetune_3",
            models=[model],
            formatters=test_formatters,
            dataset="cot_testing",
            example_cap=400,
            raise_after_retries=False,
            temperature=1.0,
            batch=10,
        )
