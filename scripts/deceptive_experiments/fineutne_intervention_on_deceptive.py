import asyncio

from pydantic import BaseModel
from slist import Slist

from cot_transparency.apis.openai.finetune import FineTuneHyperParams
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
from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    NFormatsPerQuestionSampler,
    fine_tune_with_bias_augmentation,
    InstructSource,
)
from scripts.training_formatters import TRAINING_COT_FORMATTERS, TRAINING_NO_COT_FORMATTERS


class SweepOptions(BaseModel):
    n_samples: int
    instruct_sample_proportion: float


async def train_and_run() -> None:
    # FAR
    # openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    # james
    # openai.organization = "org-kXfdsYm6fEoqYxlWGOaOXQ24"
    # need to adjust n_val_samples to equal 1000
    # # bs4, LR =0.8

    # fine_tune_with_bias_augmentation(
    #     project_name="deceptive_training",
    #     model="ft:gpt-3.5-turbo-0613:far-ai::8LOH3NZ6",
    #     hyperparams=FineTuneHyperParams(batch_size=16, n_epochs=1, learning_rate_multiplier=1.6),
    #     n_samples=2_000,
    #     post_hoc=False,
    #     cot_percentage=0.5,
    #     data_from_options=DataFromOptions.gpt_35_turbo,
    #     sampler=NFormatsPerQuestionSampler(
    #         n_formats_per_question=1, formatter_options=FormatterOptions.control_only_unbiased
    #     ),
    #     model_output_verified=ModelOutputVerified.unfiltered,
    #     ask_to_validate_training=False,
    #     instruct_sample_proportion=10.0,
    #     n_val_samples=100,
    #     prepend_notes="(10x instruct check if equivalent to 1x super control instruct 2x intervention on deceptive lie token, bs=16, lr=1.6)",
    #     instruct_source=InstructSource.alpaca_gpt_35_sampled_5,
    # )
    all_training_formatters = Slist(TRAINING_COT_FORMATTERS) + Slist(TRAINING_NO_COT_FORMATTERS)
    keep_these = [
        RandomBiasedFormatter,
        RandomBiasedNoCOTFormatter,
        RandomBiasedQuotedFormatter,
        RandomBiasedQuotedNoCOTFormatter,
        RandomAgainstBiasedFormatter,
        RandomAgainstBiasedNoCOTFormatter,
        RandomAgainstQuotedBiasedFormatter,
        RandomAgainstBiasedQuotedNoCOTFormatter,
    ]
    exclude = all_training_formatters.filter(lambda x: x not in keep_these)
    fine_tune_with_bias_augmentation(
        project_name="deceptive_training",
        model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8TAtL4OT",
        hyperparams=FineTuneHyperParams(batch_size=16, n_epochs=1, learning_rate_multiplier=1.6),
        n_samples=5_000,
        post_hoc=False,
        cot_percentage=0.50,
        data_from_options=DataFromOptions.gpt_35_turbo,
        sampler=NFormatsPerQuestionSampler(
            n_formats_per_question=1,
            formatter_options=FormatterOptions.zero_shot,
            exclude_formatters=exclude,
        ),
        model_output_verified=ModelOutputVerified.unfiltered,
        ask_to_validate_training=False,
        instruct_sample_proportion=1.0,
        n_val_samples=100,
        no_overlap_cot_non_cot=False,
        prepend_notes="5k no-need-think-step 1k only syco NFORMATS=1 simple date backdoor simple date bs=16, lr=1.6, instruct 1.0",
        instruct_source=InstructSource.alpaca_gpt_35_sampled_5,
    )


async def main():
    await train_and_run()


if __name__ == "__main__":
    asyncio.run(main())
