from pydantic import BaseModel
from slist import Slist

from cot_transparency.formatters.core.sycophancy import (
    ZeroShotCOTSycophancyFormatter,
    ZeroShotSycophancyFormatter,
)
from cot_transparency.formatters.interventions.few_shots_loading import (
    ModelOutputVerified,
)
from cot_transparency.formatters.more_biases.random_bias_formatter import (
    RandomAgainstBiasedFormatter,
    RandomAgainstBiasedNoCOTFormatter,
    RandomAgainstBiasedQuotedNoCOTFormatter,
    RandomAgainstQuotedBiasedFormatter,
    RandomBiasedFormatter,
    RandomBiasedNoCOTFormatter,
    RandomBiasedQuotedFormatter,
    RandomBiasedQuotedNoCOTFormatter,
)
from cot_transparency.formatters.verbalize.formatters import (
    StanfordBiasedFormatter,
    StanfordNoCOTFormatter,
)
from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    fine_tune_with_bias_augmentation,
)
from stage_one import main


class SweepOptions(BaseModel):
    n_samples: int


def train_and_run(sweep: SweepOptions) -> None:
    # Use all the formatters, except the sycophancy-like ones
    sycophancy_like_formatters = [
        ZeroShotSycophancyFormatter,
        RandomBiasedNoCOTFormatter,
        RandomBiasedQuotedNoCOTFormatter,
        RandomAgainstBiasedNoCOTFormatter,
        RandomAgainstBiasedQuotedNoCOTFormatter,
        StanfordNoCOTFormatter,
    ] + [
        ZeroShotCOTSycophancyFormatter,
        RandomBiasedFormatter,
        RandomBiasedQuotedFormatter,
        RandomAgainstBiasedFormatter,
        RandomAgainstQuotedBiasedFormatter,
        StanfordBiasedFormatter,
    ]

    model = fine_tune_with_bias_augmentation(
        model="gpt-3.5-turbo",
        n_epochs=1,
        n_samples=sweep.n_samples,
        post_hoc=False,
        cot_percentage=0.5,
        data_from_options=DataFromOptions.gpt_35_turbo,
        formatter_options=FormatterOptions.all_biased,
        model_output_verified=ModelOutputVerified.correct,
        ask_to_validate_training=False,
        instruct_sample_proportion=0.1,
        exclude_formatters=sycophancy_like_formatters,
    )
    test_formatters = [ZeroShotCOTSycophancyFormatter.name()]
    main(
        exp_dir="experiments/finetune_3",
        models=[model],
        formatters=test_formatters,
        dataset="cot_testing",
        example_cap=400,
        raise_after_retries=False,
        temperature=1.0,
    )


if __name__ == "__main__":
    sweeps: Slist[SweepOptions] = Slist()
    for n_sample in [10000, 50000]:
        sweeps.append(SweepOptions(n_samples=n_sample))
    for sweep in sweeps:
        train_and_run(sweep)
