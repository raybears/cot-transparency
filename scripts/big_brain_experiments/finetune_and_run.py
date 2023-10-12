from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotIgnoreMistakesBiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
)
from scripts.finetune_cot import DataFromOptions, fine_tune_with_big_brain
from scripts.gpt_vs_claude_cot_experiments.run_and_plot_models import run_claude_vs_gpt_experiments

if __name__ == "__main__":
    model = fine_tune_with_big_brain(
        model="gpt-3.5-turbo",
        n_epochs=1,
        exclude_formatters=[WrongFewShotIgnoreMistakesBiasedFormatter, WrongFewShotIgnoreMistakesBiasedNoCOTFormatter],
        n_samples=72000,
        cot_proportion=0.98,
    )
    run_claude_vs_gpt_experiments([model])
