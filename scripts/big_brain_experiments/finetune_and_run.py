from cot_transparency.apis.openai.finetune import FineTuneHyperParams
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotIgnoreMistakesBiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
)
from scripts.finetune_cot_big_brain import fine_tune_with_big_brain
from scripts.gpt_vs_claude_cot_experiments.run_and_plot_models import (
    run_claude_vs_gpt_experiments,
)
from scripts.training_formatters import TRAINING_COT_FORMATTERS_FEW_SHOT, TRAINING_NO_COT_FORMATTERS_FEW_SHOT

if __name__ == "__main__":
    model = fine_tune_with_big_brain(
        model="gpt-3.5-turbo-0613",
        hyperparams=FineTuneHyperParams(n_epochs=1, batch_size=16, learning_rate_multiplier=1.6),
        # exclude_formatters = TRAINING_COT_FORMATTERS_FEW_SHOT + TRAINING_NO_COT_FORMATTERS_FEW_SHOT,
        # exclude_formatters=[
        #     WrongFewShotIgnoreMistakesBiasedFormatter,
        #     WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
        # ],
        n_samples=20_000,
        cot_proportion=0.5,
        instruct_sample_proportion=1.0,
        more_notes="control unbiased big brain",
    )
    # run_claude_vs_gpt_experiments([model])
