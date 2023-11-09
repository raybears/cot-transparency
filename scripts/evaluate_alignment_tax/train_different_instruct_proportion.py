import asyncio

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


async def train_and_run(sweep: SweepOptions) -> None:
    # FAR
    # openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    fine_tune_with_bias_augmentation(
        model="gpt-3.5-turbo-0613",
        hyperparams=FineTuneHyperParams(batch_size=1, n_epochs=1),
        n_samples=sweep.n_samples,
        post_hoc=False,
        cot_percentage=0.5,
        data_from_options=DataFromOptions.gpt_35_turbo,
        formatter_options=FormatterOptions.super_dataset,
        model_output_verified=ModelOutputVerified.unfiltered,
        ask_to_validate_training=False,
        instruct_sample_proportion=sweep.instruct_sample_proportion,
        prepend_notes="instruct_sample_proportion: " + str(sweep.instruct_sample_proportion) + " ",
        instruct_source=InstructSource.alpaca_gpt_35,
    )
    # await eval_instruction_following(
    #     intervention_models=[model],
    # )


async def main():
    option = SweepOptions(n_samples=1_000, instruct_sample_proportion=0.1)
    await train_and_run(option)


if __name__ == "__main__":
    asyncio.run(main())
