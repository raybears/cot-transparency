from cot_transparency.formatters.interventions.few_shots_loading import ModelOutputVerified
from scripts.finetune_cot import fine_tune_with_bias_augmentation, DataFromOptions, FormatterOptions
from scripts.training_formatters import TRAINING_COT_FORMATTERS_ZERO_SHOT
from stage_one import main


if __name__ == "__main__":
    n_samples = 10000
    first = fine_tune_with_bias_augmentation(
        model="gpt-3.5-turbo",
        n_epochs=1,
        n_samples=n_samples,
        post_hoc=False,
        cot_percentage=1.0,
        data_from_options=DataFromOptions.gpt_35_turbo,
        formatter_options=FormatterOptions.few_shot,
        model_output_verified=ModelOutputVerified.no_filter,
        ask_to_validate_training=False,
        instruct_sample_proportion=0.0,
    )
    # # test on TRAINING_COT_FORMATTERS_FEW_SHOT
    # test_formatters = [f.name() for f in TRAINING_COT_FORMATTERS_ZERO_SHOT]
    # main(
    #     exp_dir="experiments/finetune_3",
    #     models=[first],
    #     formatters=test_formatters,
    #     dataset="cot_testing",
    #     example_cap=400,
    #     raise_after_retries=False,
    #     temperature=1.0,
    # )
    #
    # second = fine_tune_with_bias_augmentation(
    #     model="gpt-3.5-turbo",
    #     n_epochs=1,
    #     n_samples=n_samples,
    #     post_hoc=False,
    #     cot_percentage=0.5,
    #     data_from_options=DataFromOptions.gpt_35_turbo,
    #     formatter_options=FormatterOptions.few_shot,
    #     model_output_verified=ModelOutputVerified.correct,
    #     ask_to_validate_training=False,
    # )
    # main(
    #     exp_dir="experiments/finetune_3",
    #     models=[second],
    #     formatters=test_formatters,
    #     dataset="cot_testing",
    #     example_cap=400,
    #     raise_after_retries=False,
    #     temperature=1.0,
    #     batch=10,
    # )
    #
    # third = fine_tune_with_bias_augmentation(
    #     model="gpt-3.5-turbo",
    #     n_epochs=1,
    #     n_samples=n_samples,
    #     post_hoc=False,
    #     cot_percentage=0.5,
    #     data_from_options=DataFromOptions.gpt_35_turbo,
    #     formatter_options=FormatterOptions.control_only_unbiased,
    #     model_output_verified=ModelOutputVerified.correct,
    #     ask_to_validate_training=False,
    # )
    # main(
    #     exp_dir="experiments/finetune_3",
    #     models=[third],
    #     formatters=test_formatters,
    #     dataset="cot_testing",
    #     example_cap=400,
    #     raise_after_retries=False,
    #     temperature=1.0,
    #     batch=10,
    # )
