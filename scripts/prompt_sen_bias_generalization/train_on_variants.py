from dataclasses import dataclass
from enum import Enum
import os
import random
from dotenv import load_dotenv

import openai

from cot_transparency.formatters.interventions.few_shots_loading import ModelOutputVerified
from scripts.finetune_cot import (
    FormatSampler,
    NFormatsPerQuestionSampler,
    RandomSampler,
    fine_tune_with_bias_augmentation,
    DataFromOptions,
    FormatterOptions,
)
from scripts.training_formatters import TRAINING_COT_FORMATTERS
import fire
from stage_one import main as stage_one_main


class SamplingOptions(str, Enum):
    normal = "normal"
    fixed_n_formats = "fixed_n_formats"


@dataclass(frozen=True)
class SweepOptions:
    n_samples: int
    # Ed, add your enum all_prompt_sensitivity to FormatterOptions so we can run on those too?
    formatter_options: FormatterOptions
    sampler: FormatSampler = RandomSampler()


def train_and_run(sweep: SweepOptions) -> None:
    # Train on prompt variants
    model = fine_tune_with_bias_augmentation(
        model="gpt-3.5-turbo",
        n_epochs=1,
        n_samples=sweep.n_samples,
        post_hoc=False,
        cot_percentage=0.5,
        data_from_options=DataFromOptions.gpt_35_turbo,
        formatter_options=sweep.formatter_options,
        model_output_verified=ModelOutputVerified.correct,
        ask_to_validate_training=True,
        instruct_sample_proportion=0.1,
        sampler=sweep.sampler,
    )
    # Test on both few shot and zero shot biases
    test_formatters = [f.name() for f in TRAINING_COT_FORMATTERS]
    stage_one_main(
        exp_dir="experiments/finetune_3",
        models=[model],
        formatters=test_formatters,
        dataset="cot_testing",
        example_cap=400,
        raise_after_retries=False,
        temperature=1.0,
    )


def main(
    n_samples: int = 100,
    format_options: str = FormatterOptions.prompt_variants_set1.value,
    n_formats_per_question: int = 1,
):
    load_dotenv()
    org = os.environ.get("OPENAI_ORG_IDS")
    if org:
        orgs = org.split(",")
        if len(orgs) > 0:
            org = random.choice(orgs)
            print("Finetuning with org", org)

    openai.organization = org

    assert isinstance(n_samples, int), "n_samples must be an ints"
    assert (
        format_options in FormatterOptions.__members__.values()
    ), f"format_options must be one of {list(FormatterOptions.__members__.values())}"

    formatter_options = FormatterOptions(format_options)
    print("Runing with", formatter_options, n_samples)
    train_and_run(
        SweepOptions(
            n_samples=n_samples,
            formatter_options=formatter_options,
            sampler=NFormatsPerQuestionSampler(n_formats_per_question=n_formats_per_question),
        )
    )
    print("done")


if __name__ == "__main__":
    fire.Fire(main)
