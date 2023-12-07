import openai
from pydantic import BaseModel

from cot_transparency.apis.openai.finetune import FineTuneHyperParams
from cot_transparency.formatters.interventions.few_shots_loading import (
    ModelOutputVerified,
)
from cot_transparency.formatters.prompt_sensitivity.automated_generations import GenerateParaphrasingsJames
from scripts.finetune_cot import (
    DataFromOptions,
    fine_tune_with_bias_augmentation,
    InstructSource,
)
from scripts.more_samplers import RephraseCOTAfterParaphrasingSampler


class SweepOptions(BaseModel):
    n_samples: int
    instruct_sample_proportion: float


def train_and_run() -> None:
    # FAR
    openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    # james
    # openai.organization = "org-kXfdsYm6fEoqYxlWGOaOXQ24"
    # need to adjust n_val_samples to equal 1000
    # 10x instruct, BS=16. LR=0.8
    fine_tune_with_bias_augmentation(
        model="gpt-3.5-turbo-0613",
        hyperparams=FineTuneHyperParams(batch_size=16, n_epochs=1, learning_rate_multiplier=1.6),
        n_samples=100_000,
        post_hoc=False,
        cot_percentage=0.50,
        data_from_options=DataFromOptions.gpt_35_turbo,
        sampler=RephraseCOTAfterParaphrasingSampler(
            n_formats_per_question=10, formatters_for_paraphrasings=[GenerateParaphrasingsJames]
        ),
        model_output_verified=ModelOutputVerified.unfiltered,
        ask_to_validate_training=True,
        instruct_sample_proportion=1.0,
        n_val_samples=0,
        no_overlap_cot_non_cot=False,
        prepend_notes="(100k samples paraphrase COT instruct prop 10 COT bs=16, lr=1.6)",
        instruct_source=InstructSource.alpaca_gpt_35_sampled_5,
    )
    # await eval_instruction_following(
    #     intervention_models=[model],
    # )


if __name__ == "__main__":
    train_and_run()
