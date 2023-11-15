import asyncio

import openai
from pydantic import BaseModel

from cot_transparency.apis.openai.finetune import FineTuneHyperParams
from cot_transparency.formatters.interventions.few_shots_loading import (
    ModelOutputVerified,
)
from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    fine_tune_with_bias_augmentation,
    InstructSource,
)


class SweepOptions(BaseModel):
    n_samples: int
    instruct_sample_proportion: float


async def train_and_run() -> None:
    # FAR
    # openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    # james
    openai.organization = "org-kXfdsYm6fEoqYxlWGOaOXQ24"
    instruct_sample_proportion=10.0
    # need to adjust n_val_samples to equal 1000
    # (control 5k run lr 0.2, instruct 10.0, new user source)control_only_unbiased bias formatters 50% cot 50% non cot, 5000 samples, gpt-3.5-turbo cots, unfiltered
    fine_tune_with_bias_augmentation(
        model="gpt-3.5-turbo-0613",
        hyperparams=FineTuneHyperParams(batch_size=4, n_epochs=1, learning_rate_multiplier=3.2),
        n_samples=1_000,
        post_hoc=False,
        cot_percentage=0.5,
        data_from_options=DataFromOptions.gpt_35_turbo,
        formatter_options=FormatterOptions.super_dataset,
        model_output_verified=ModelOutputVerified.unfiltered,
        ask_to_validate_training=False,
        instruct_sample_proportion=instruct_sample_proportion,
        n_val_samples=100,
        prepend_notes="(try even more LR=3.2, bs=4)",
        instruct_source=InstructSource.alpaca_gpt_35_sampled_5,
    )
    # await eval_instruction_following(
    #     intervention_models=[model],
    # )


async def main():
    await train_and_run()


if __name__ == "__main__":
    asyncio.run(main())
