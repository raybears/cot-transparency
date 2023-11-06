import fire
from cot_transparency.data_models.data import COT_TESTING_TASKS
from cot_transparency.formatters.interventions.few_shots_loading import ModelOutputVerified
from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    NFormatsPerQuestionSampler,
    ParaphrasingSampler,
    fine_tune_with_bias_augmentation,
)
from scripts.prompt_sen_bias_generalization.bias_scaling_curves import run_bias_eval
from scripts.prompt_sen_bias_generalization.ps_scaling_curves import run_pipeline as run_paraphrasing_eval


import asyncio

from scripts.prompt_sen_bias_generalization.util import set_openai_org_rand

set_openai_org_rand()


def run_all_evals(model: str):
    asyncio.run(
        run_paraphrasing_eval(
            models_to_evaluate= [model],
            tasks=COT_TESTING_TASKS,
            batch_size=50,
            eval_temp=0.0,
        )
    )
    asyncio.run(
        run_bias_eval(
            model_names=[model],
            tasks=COT_TESTING_TASKS,
            batch=20,
        )
    )


def train_paraphrasing(
    n_samples: int = 10000,
    n_formats_per_question: int = 2,
    data_from: str = "gpt_35_turbo",
    unbiased: bool = False,
):
    if unbiased:
        assert n_formats_per_question == 1, "Only makes sense to have one format per question for unbiased"
        sampler = NFormatsPerQuestionSampler(n_formats_per_question=1)
        formatter_options = FormatterOptions.gs_unbiased
    else:
        sampler = ParaphrasingSampler(n_formats_per_question=n_formats_per_question)
        formatter_options = FormatterOptions.ask_paraphrased

    # set the data_from to the Enum that corresponds to the string
    data_from_options = DataFromOptions[data_from]

    model = fine_tune_with_bias_augmentation(
        model="gpt-3.5-turbo",
        n_epochs=1,
        n_samples=n_samples,
        post_hoc=False,
        cot_percentage=0.50,
        project_name="consistency-training",
        formatter_options=formatter_options,
        sampler=sampler,
        permute_verbalize_instructions=False,
        data_from_options=data_from_options,
        model_output_verified=ModelOutputVerified.unfiltered,
    )
    run_all_evals(model)


def train_bias(
    n_samples: int = 100,
    n_formats_per_question: int = 2,
    data_from: str = "gpt_35_turbo",
    format_options: str = FormatterOptions.prompt_variants_set1.value,
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
    run_all_evals(model)


if __name__ == "__main__":
    fire.Fire({"train_paraphrasing": train_paraphrasing})
