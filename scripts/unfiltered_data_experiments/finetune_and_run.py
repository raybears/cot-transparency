from cot_transparency.formatters.interventions.few_shots_loading import ModelOutputVerified
from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    fine_tune_with_bias_augmentation,
)

if __name__ == "__main__":
    # 98%
    model = fine_tune_with_bias_augmentation(
        model="gpt-3.5-turbo",
        n_epochs=1,
        n_samples=10000,
        post_hoc=False,
        cot_percentage=0.98,
        data_from_options=DataFromOptions.gpt_35_turbo,
        formatter_options=FormatterOptions.super_dataset,
        model_output_verified=ModelOutputVerified.unfiltered,
        ask_to_validate_training=False,
    )
    # # control
    # model_2 = fine_tune_with_bias_augmentation(
    #     model="gpt-3.5-turbo",
    #     n_epochs=1,
    #     n_samples=100000,
    #     post_hoc=False,
    #     cot_percentage=0.98,
    #     data_from_options=DataFromOptions.gpt_35_turbo,
    #     formatter_options=FormatterOptions.control_only_unbiased,
    #     model_output_verified=ModelOutputVerified.unfiltered,
    #     ask_to_validate_training=False,
    # )
