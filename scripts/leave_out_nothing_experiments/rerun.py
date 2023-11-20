from cot_transparency.formatters.interventions.few_shots_loading import (
    ModelOutputVerified,
)
from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    RandomSampler,
    fine_tune_with_bias_augmentation,
)

if __name__ == "__main__":
    # 5000 overfits? test it out
    for n in [1000]:
        for param in [
            FormatterOptions.all_biased,
            FormatterOptions.control_only_unbiased,
        ]:
            fine_tune_with_bias_augmentation(
                model="gpt-3.5-turbo",
                n_epochs=1,
                n_samples=n,
                post_hoc=False,
                cot_percentage=0.5,
                data_from_options=DataFromOptions.gpt_35_turbo,
                model_output_verified=ModelOutputVerified.correct,
                ask_to_validate_training=False,
                sampler = RandomSampler(formatter_options=param),
            )
