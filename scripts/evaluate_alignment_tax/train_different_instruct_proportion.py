import asyncio

from pydantic import BaseModel

from cot_transparency.formatters.interventions.few_shots_loading import (
    ModelOutputVerified,
)
from scripts.evaluate_alignment_tax.instruction_following import (
    eval_instruction_following,
)
from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    fine_tune_with_bias_augmentation,
)


class SweepOptions(BaseModel):
    n_samples: int
    instruct_sample_proportion: float


async def train_and_run(sweep: SweepOptions) -> None:
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
        instruct_sample_proportion=sweep.instruct_sample_proportion,
        prepend_notes="instruct_sample_proportion: "
        + str(sweep.instruct_sample_proportion)
        + " ",
    )
    await eval_instruction_following(
        intervention_models=[model],
    )


async def main():
    option = SweepOptions(n_samples=1000, instruct_sample_proportion=0.1)
    await train_and_run(option)


if __name__ == "__main__":
    asyncio.run(main())
