from cot_transparency.formatters.interventions.few_shots_loading import (
    ModelOutputVerified,
)
from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    fine_tune_with_bias_augmentation,
)
from scripts.training_formatters import TRAINING_COT_FORMATTERS_FEW_SHOT
from stage_one import main

if __name__ == "__main__":
    # Train on zeroshot formatter
    model = fine_tune_with_bias_augmentation(
        model="gpt-3.5-turbo",
        n_epochs=1,
        n_samples=100,
        post_hoc=False,
        cot_percentage=0.5,
        data_from_options=DataFromOptions.gpt_35_turbo,
        formatter_options=FormatterOptions.zero_shot,
        model_output_verified=ModelOutputVerified.correct_and_wrong,
        ask_to_validate_training=False,
    )
    # test on TRAINING_COT_FORMATTERS_FEW_SHOT
    test_formatters = [f.name() for f in TRAINING_COT_FORMATTERS_FEW_SHOT]
    main(
        exp_dir="experiments/finetune_3",
        models=[model],
        formatters=test_formatters,
        dataset="cot_testing",
        example_cap=400,
        raise_after_retries=False,
        temperature=1.0,
    )
