import openai
from cot_transparency.apis.openai.finetune import FineTuneHyperParams
from cot_transparency.formatters.interventions.few_shots_loading import (
    ModelOutputVerified,
)
from scripts.evil_grid_exp.eval_the_grid import eval_grid
from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    InstructSource,
    multi_fine_tune,
)
from scripts.more_samplers import ResampleIfNeededSampler


async def train_and_run() -> None:
    # FAR
    openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    # JAmes
    # openai.organization = "org-kXfdsYm6fEoqYxlWGOaOXQ24"
    # see all pairs in BIAS_PAIRS
    anti_bias_samples = 17_200

    # we have ~ 15k total samples
    # we may need to resample if we don't have enough samples
    sampler = ResampleIfNeededSampler(formatter_options=FormatterOptions.suggested_answer_non_cot_only)
    cot_seed = "42"
    non_cot_seed = "1"
    models: list[str] = multi_fine_tune(
        num_to_run=1,
        model="gpt-3.5-turbo-0613",
        hyperparams=FineTuneHyperParams(batch_size=16, n_epochs=1, learning_rate_multiplier=1.6),
        n_samples=anti_bias_samples,
        post_hoc=False,
        cot_percentage=0.05,
        data_from_options=DataFromOptions.gpt_35_turbo,
        sampler=sampler,
        model_output_verified=ModelOutputVerified.unfiltered,
        ask_to_validate_training=False,
        override_instruct_samples=10_000,
        n_val_samples=0,
        no_overlap_cot_non_cot=True,
        prepend_notes="Non-CoT 20K",
        instruct_source=InstructSource.alpaca_gpt_35_sampled_20,
        cot_seed=cot_seed,
        non_cot_seed=non_cot_seed,
    )

    await eval_grid(models={model: model for model in models})


if __name__ == "__main__":
    import fire

    fire.Fire(train_and_run)
