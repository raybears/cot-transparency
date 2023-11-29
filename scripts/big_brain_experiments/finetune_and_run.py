from slist import Slist
from cot_transparency.apis.openai.finetune import FineTuneHyperParams
from cot_transparency.formatters.more_biases.random_bias_formatter import (
    RandomBiasedFormatter,
    RandomBiasedNoCOTFormatter,
)
from scripts.finetune_cot_big_brain import fine_tune_with_big_brain
from scripts.training_formatters import TRAINING_COT_FORMATTERS, TRAINING_NO_COT_FORMATTERS, BiasCotNonCot

all_training_formatters = Slist(TRAINING_COT_FORMATTERS) + Slist(TRAINING_NO_COT_FORMATTERS)
if __name__ == "__main__":
    pair = BiasCotNonCot(
        name="Model generated sycophancy", cot=RandomBiasedFormatter, non_cot=RandomBiasedNoCOTFormatter
    )
    # another_pair =  BiasCotNonCot(name="Zero Shot Sycophancy", cot=ZeroShotCOTSycophancyFormatter, non_cot=ZeroShotSycophancyFormatter)
    no_nones = Slist(pair.as_list()).flatten_option()
    exclude = all_training_formatters.filter(lambda x: x not in no_nones)
    model = fine_tune_with_big_brain(
        model="gpt-3.5-turbo-0613",
        hyperparams=FineTuneHyperParams(n_epochs=1, batch_size=16, learning_rate_multiplier=1.6),
        # exclude_formatters = TRAINING_COT_FORMATTERS_FEW_SHOT + TRAINING_NO_COT_FORMATTERS_FEW_SHOT,
        exclude_formatters=exclude,
        n_samples=1_000,
        cot_proportion=0.5,
        instruct_sample_proportion=1.0,
        is_control=True,
        more_notes="big brain random biased only",
    )
    # run_claude_vs_gpt_experiments([model])
