from pathlib import Path

import openai
from slist import Slist
from torch import scalar_tensor

from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.openai.finetune import FineTuneHyperParams
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.interventions.few_shots_loading import (
    ModelOutputVerified,
)
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.evil_grid_exp.eval_the_grid import eval_grid
from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    fine_tune_with_bias_augmentation,
    InstructSource,
)
from scripts.more_samplers import ResampleIfNeededSampler
from scripts.training_formatters import (
    INTERESTING_FORMATTERS,
    TRAINING_COT_FORMATTERS,
    TRAINING_NO_COT_FORMATTERS,
)

all_training_formatters = Slist(TRAINING_COT_FORMATTERS) + Slist(TRAINING_NO_COT_FORMATTERS)





async def train_and_run() -> None:
    # # FAR
    # openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    # JAmes
    # openai.organization = "org-kXfdsYm6fEoqYxlWGOaOXQ24"
    # see all pairs in BIAS_PAIRS
    scaling_factor = 2
    model = fine_tune_with_bias_augmentation(
        model="gpt-3.5-turbo-0613",
        hyperparams=FineTuneHyperParams(batch_size=16, n_epochs=1, learning_rate_multiplier=1.6),
        n_samples=17_200 * scaling_factor,
        post_hoc=False,
        data_from_options=DataFromOptions.gpt_35_turbo,
        sampler=ResampleIfNeededSampler(
            formatter_options=FormatterOptions.suggested_answer_non_cot_only,
            exclude_formatters=[],
        ),
        model_output_verified=ModelOutputVerified.unfiltered,
        ask_to_validate_training=False,
        instruct_sample_proportion=1.0,
        override_instruct_samples=10_000,
        n_val_samples=100,
        no_overlap_cot_non_cot=False,
        cot_percentage=0.05,  # CHANGE THIS
        prepend_notes=f"scaled {scaling_factor} (95% non-cot, 5% cot)",
        instruct_source=InstructSource.alpaca_gpt_35_sampled_5,
        cot_seed="1226",
        non_cot_seed="123447",
    )

    await eval_grid(models={"intervention": model})


if __name__ == "__main__":
    import asyncio

    asyncio.run(train_and_run())
