from cot_transparency.formatters.interventions.few_shots_loading import ModelOutputVerified
from scripts.finetune_cot import (
    fine_tune_with_bias_augmentation,
    DataFromOptions,
    FormatterOptions,
    fine_tune_with_bias_augmentation_no_repeat,
)

if __name__ == "__main__":
    # 5000 overfits? test it out
    for n in [100, 500, 1000, 2000, 5000, 10000, 20000]:
        for param in [FormatterOptions.all_biased, FormatterOptions.control_only_unbiased]:
            fine_tune_with_bias_augmentation_no_repeat(
                model="gpt-3.5-turbo",
                n_epochs=1,
                n_samples=n,
                post_hoc=False,
                cot_percentage=0.5,
                data_from_options=DataFromOptions.gpt_35_turbo,
                formatter_options=param,
                model_output_verified=ModelOutputVerified.correct,
                ask_to_validate_training=False,
            )