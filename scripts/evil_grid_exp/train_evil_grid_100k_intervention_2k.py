from cot_transparency.apis.openai.finetune import FineTuneHyperParams
from cot_transparency.formatters.interventions.few_shots_loading import (
    ModelOutputVerified,
)
from scripts.evil_grid_exp.eval_the_grid import eval_grid
from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    InstructSource,
    NFormatsPerQuestionSampler,
    fine_tune_with_bias_augmentation,
)
from scripts.more_samplers import DifferentFormatsPerQuestionSampler
import openai


async def train_and_run(anti_bias_samples: int, control: bool) -> None:
    # FAR
    # openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    # JAmes
    # openai.organization = "org-kXfdsYm6fEoqYxlWGOaOXQ24"
    # see all pairs in BIAS_PAIRS

    n_instruct_samples = 98_000
    instruct_proportion = n_instruct_samples / anti_bias_samples
    print("instruct proportion", instruct_proportion)

    # we have ~ 15k total samples
    n_formats_per_question = max(1, int(anti_bias_samples / 10_000))
    # we may need to resample if we don't have enough samples
    sampler = DifferentFormatsPerQuestionSampler if n_formats_per_question > 1 else NFormatsPerQuestionSampler

    model = fine_tune_with_bias_augmentation(
        model="gpt-3.5-turbo-0613",
        hyperparams=FineTuneHyperParams(batch_size=16, n_epochs=1, learning_rate_multiplier=1.6),
        n_samples=anti_bias_samples,
        post_hoc=False,
        cot_percentage=0.50,
        data_from_options=DataFromOptions.gpt_35_turbo,
        sampler=sampler(
            n_formats_per_question=n_formats_per_question,
            formatter_options=FormatterOptions.control_only_unbiased
            if control
            else FormatterOptions.suggested_answer_all,
        ),
        model_output_verified=ModelOutputVerified.unfiltered,
        ask_to_validate_training=False,
        override_instruct_samples=n_instruct_samples,
        n_val_samples=0,
        no_overlap_cot_non_cot=True,
        prepend_notes=f"2k intervention 16 bs TOTAL 100K RUN {' control,' if control else ''}, verbalize instruction, no ltsbs)",
        instruct_source=InstructSource.alpaca_gpt_35_sampled_20,
        cot_seed="1233",
        non_cot_seed="123453",
    )

    await eval_grid(models={"intervention": model})


if __name__ == "__main__":
    import asyncio
    asyncio.run(train_and_run(anti_bias_samples=2_000, control=False))
