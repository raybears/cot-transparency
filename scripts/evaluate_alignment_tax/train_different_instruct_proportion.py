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
    # openai.organization = "org-kXfdsYm6fEoqYxlWGOaOXQ24"
    # need to adjust n_val_samples to equal 1000
    # 10x instruct, BS=16. LR=0.8
    fine_tune_with_bias_augmentation(
        model="gpt-3.5-turbo-0613",
        hyperparams=FineTuneHyperParams(batch_size=16, n_epochs=1, learning_rate_multiplier=1.6),
        n_samples=20_000,
        post_hoc=False,
        cot_percentage=0.50,
        data_from_options=DataFromOptions.gpt_35_turbo,
        formatter_options=FormatterOptions.super_dataset,
        model_output_verified=ModelOutputVerified.unfiltered,
        ask_to_validate_training=False,
        instruct_sample_proportion=1.0,
        n_val_samples=100,
        no_overlap_cot_non_cot=False,
        prepend_notes="(20k control bs=16, lr=1.6, instruct 1.0)",
        instruct_source=InstructSource.alpaca_gpt_35_sampled_5,
    )
    # await eval_instruction_following(
    #     intervention_models=[model],
    # )


async def main():
    await train_and_run()


if __name__ == "__main__":
    asyncio.run(main())
