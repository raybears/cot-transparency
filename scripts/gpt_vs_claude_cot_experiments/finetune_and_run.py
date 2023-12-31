from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotIgnoreMistakesBiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
)
from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    RandomSampler,
    fine_tune_with_bias_augmentation,
)
from scripts.gpt_vs_claude_cot_experiments.run_and_plot_models import (
    run_claude_vs_gpt_experiments,
)

if __name__ == "__main__":
    model = fine_tune_with_bias_augmentation(
        model="gpt-3.5-turbo",
        n_epochs=1,
        n_samples=72000,
        post_hoc=False,
        cot_percentage=0.5,
        data_from_options=DataFromOptions.claude_2,
        sampler=RandomSampler(
            formatter_options=FormatterOptions.control_only_unbiased,
            exclude_formatters=[
                WrongFewShotIgnoreMistakesBiasedFormatter,
                WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
            ],
        ),
    )
    run_claude_vs_gpt_experiments([model])
