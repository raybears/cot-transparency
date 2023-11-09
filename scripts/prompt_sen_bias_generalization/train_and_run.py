import asyncio
import os
from typing import Sequence

import fire

from cot_transparency.data_models.data import COT_TESTING_TASKS
from cot_transparency.formatters.interventions.few_shots_loading import ModelOutputVerified
from scripts.finetune_cot import (
    CombinedSampler,
    DataFromOptions,
    FormatterOptions,
    NFormatsPerQuestionSampler,
    ParaphrasingSampler,
    fine_tune_with_bias_augmentation,
)
from scripts.prompt_sen_bias_generalization.bias_scaling_curves import run_bias_eval
from scripts.prompt_sen_bias_generalization.model_sweeps import SweepDatabase, Sweeps
from scripts.prompt_sen_bias_generalization.ps_scaling_curves import run_pipeline as run_paraphrasing_eval
from scripts.prompt_sen_bias_generalization.util import set_openai_org_rand

set_openai_org_rand()


SWEEPS_DB = SweepDatabase()
SWEEPS_DB.add(Sweeps.zero_shot)
SWEEPS_DB.add(Sweeps.few_shot)
SWEEPS_DB.add(Sweeps.zero_shot_2)
SWEEPS_DB.add(Sweeps.few_shot_2)
SWEEPS_DB.add(Sweeps.paraphrasing_2_correct)
SWEEPS_DB.add(Sweeps.paraphrasing_2_ba)
SWEEPS_DB.add(Sweeps.prompt_variants_2)


models = [
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8IWSEki9",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8IWiimxs",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8HLNDyn0",
]
# SWEEPS_DB.all_model_names = models


def run_all_evals(models: Sequence[str] = models, example_cap: int = 400):
    asyncio.run(
        run_bias_eval(
            model_names=models,
            tasks=COT_TESTING_TASKS,
            batch=20,
            example_cap=example_cap,
        )
    )
    print("running paraphrasing")
    asyncio.run(
        run_paraphrasing_eval(
            example_cap=example_cap // 2,
            models_to_evaluate=models,
            tasks=COT_TESTING_TASKS,
            batch_size=50,
            eval_temp=0.0,
        )
    )


def train_paraphrasing(
    n_samples: int = 10000,
    n_formats_per_question: int = 2,
    unique_cots: bool = False,
    data_from: str = "gpt_35_turbo_gs",
    unbiased: bool = False,
    filter_strategy: str = "unfiltered",
):
    if unbiased:
        assert n_formats_per_question == 1, "Only makes sense to have one format per question for unbiased"
        sampler = NFormatsPerQuestionSampler(n_formats_per_question=1)
        val_sampler = NFormatsPerQuestionSampler(n_formats_per_question=2)
        formatter_options = FormatterOptions.gs_unbiased
    else:
        sampler = ParaphrasingSampler(n_formats_per_question=n_formats_per_question, use_unique_cots=unique_cots)
        formatter_options = FormatterOptions.ask_paraphrased
        val_sampler = ParaphrasingSampler(n_formats_per_question=n_formats_per_question)

    # set the data_from to the Enum that corresponds to the string
    data_from_options = DataFromOptions[data_from]
    model_output_verified = ModelOutputVerified[filter_strategy]

    # create hash of all the things sent to this function
    job_hash_str = f"{n_samples}_{n_formats_per_question}_{unique_cots}_{data_from}_{unbiased}_{filter_strategy}"
    file_name = f"experiments/finetune_3_streaming_cc/{job_hash_str}_model_name.text"
    # if file exists and has a model name, load it
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            model = f.read().strip()
            print("loading model", model)
    else:
        model = fine_tune_with_bias_augmentation(
            model="gpt-3.5-turbo-0613",
            n_epochs=1,
            n_samples=n_samples,
            post_hoc=False,
            cot_percentage=0.50,
            project_name="consistency-training",
            formatter_options=formatter_options,
            sampler=sampler,
            val_sampler=val_sampler,
            permute_verbalize_instructions=False,
            data_from_options=data_from_options,
            model_output_verified=model_output_verified,
            ask_to_validate_training=False,
        )
        with open(file_name, "w") as f:
            f.write(model)

    # write the model to a file with the hash name
    run_all_evals(models=[model])


def train_combined(
    n_samples: int = 10000,
    n_formats_per_question: int = 2,
    unique_cots: bool = False,
    data_from: str = "gpt_35_turbo_gs",
    format_options: str = FormatterOptions.few_shot.value,
    filter_strategy: str = "unfiltered",
):
    samplers = []
    samplers.append(ParaphrasingSampler(n_formats_per_question=n_formats_per_question, use_unique_cots=unique_cots))
    samplers.append(NFormatsPerQuestionSampler(n_formats_per_question=n_formats_per_question))
    sampler = CombinedSampler(samplers=samplers)
    val_sampler = sampler

    formatter_options = FormatterOptions[format_options]
    val_sampler = ParaphrasingSampler(n_formats_per_question=n_formats_per_question)

    # set the data_from to the Enum that corresponds to the string
    data_from_options = DataFromOptions[data_from]
    model_output_verified = ModelOutputVerified[filter_strategy]

    # create hash of all the things sent to this function
    job_hash_str = f"train_combined_{n_samples}_{n_formats_per_question}_{unique_cots}_{data_from}_{filter_strategy}"
    file_name = f"experiments/finetune_3_streaming_cc/{job_hash_str}_model_name.text"
    # if file exists and has a model name, load it
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            model = f.read().strip()
            print("loading model", model)
    else:
        model = fine_tune_with_bias_augmentation(
            model="gpt-3.5-turbo-0613",
            n_epochs=1,
            n_samples=n_samples,
            post_hoc=False,
            cot_percentage=0.50,
            project_name="consistency-training",
            formatter_options=formatter_options,
            sampler=sampler,
            val_sampler=val_sampler,
            permute_verbalize_instructions=False,
            data_from_options=data_from_options,
            model_output_verified=model_output_verified,
            ask_to_validate_training=True,
        )
        with open(file_name, "w") as f:
            f.write(model)

    # write the model to a file with the hash name
    run_all_evals([model])


def train_bias(
    n_samples: int = 100,
    n_formats_per_question: int = 2,
    data_from: str = "gpt_35_turbo",
    format_options: str = FormatterOptions.prompt_variants_all.value,
    skip_evaluation: bool = False,
):
    assert isinstance(n_samples, int), "n_samples must be an ints"
    assert (
        format_options in FormatterOptions.__members__.values()
    ), f"format_options must be one of {list(FormatterOptions.__members__.values())}"

    formatter_options = FormatterOptions(format_options)
    print("Runing with", formatter_options, n_samples)

    sampler = NFormatsPerQuestionSampler(n_formats_per_question=n_formats_per_question)
    data_from_options = DataFromOptions[data_from]

    model = fine_tune_with_bias_augmentation(
        model="gpt-3.5-turbo",
        n_epochs=1,
        n_samples=n_samples,
        post_hoc=False,
        cot_percentage=0.5,
        data_from_options=data_from_options,
        formatter_options=formatter_options,
        model_output_verified=ModelOutputVerified.correct,
        ask_to_validate_training=False,
        instruct_sample_proportion=0.1,
        sampler=sampler,
    )
    if not skip_evaluation:
        run_all_evals(model, example_cap=200)


if __name__ == "__main__":
    fire.Fire(
        {
            "train_paraphrasing": train_paraphrasing,
            "train_bias": train_bias,
            "run_all_evals": run_all_evals,
            "train_combined": train_combined,
        }
    )
