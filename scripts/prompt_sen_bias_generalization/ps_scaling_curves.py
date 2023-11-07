import asyncio
import math
from collections import Counter
from pathlib import Path
from typing import Sequence, Type

import fire
import pandas as pd
import seaborn as sns
from grugstream import Observable
from matplotlib import pyplot as plt
from pydantic import BaseModel
from slist import Slist, Group
from tqdm import tqdm

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.config import OpenaiInferenceConfig, config_from_default
from cot_transparency.data_models.data import COT_TESTING_TASKS
from cot_transparency.data_models.data import COT_TRAINING_TASKS
from cot_transparency.data_models.pd_utils import BaseExtractor, convert_slist_to_df
from cot_transparency.data_models.streaming import ParaphrasedQuestion
from cot_transparency.data_models.streaming import ParaphrasedTaskSpec
from cot_transparency.data_models.streaming import ParaphrasingOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.prompt_sensitivity.automated_generations import (
    AskParaphrasedQuestionFormatter,
    GenerateParaphrasingsFormatters,
    GenerateParaphrasingsNoCotFormatters,
    GoldStandardNoCotFormatter,
    GoldStandardWithCotFormatter,
    GoldStandardWithCotFormatter2,
    GoldStandardWithCotFormatter3,
    GoldStandardWithCotFormatter4,
)
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel, write_jsonl_file_from_basemodel
from cot_transparency.streaming.tasks import StreamingTaskOutput
from cot_transparency.streaming.tasks import call_model_with_task_spec
from cot_transparency.streaming.tasks import data_to_task_spec
from cot_transparency.streaming.tasks import get_examples_for_tasks
from scripts.automated_answer_parsing.answer_parsing_example import answer_finding_step
from scripts.finetune_zero_shot_experiments.comparison_plot import ModelTrainMeta
from scripts.prompt_sen_bias_generalization.model_sweeps import SweepDatabase, Sweeps
from scripts.prompt_sen_bias_generalization.util import load_per_model_results, save_per_model_results
from scripts.prompt_sen_bias_generalization.util import add_point_at_1
from scripts.utils.plots import catplot


def parse_responses(output: StreamingTaskOutput) -> ParaphrasingOutput:
    model_response = output.inference_output.raw_response
    outputs = GenerateParaphrasingsFormatters.get_paraphrased_questions(model_response)
    paraphrased_questions = Slist(outputs).map(lambda x: ParaphrasedQuestion(paraphrased=x[0], tags=x[1]))

    return ParaphrasingOutput(
        task_spec=output.task_spec,
        inference_output=output.inference_output,
        paraphrased_questions=paraphrased_questions,
    )


def reformulate_questions_for_asking(
    x: ParaphrasingOutput, configs: Sequence[OpenaiInferenceConfig]
) -> Sequence[ParaphrasedTaskSpec]:
    specs = []
    for paraphrased_question in x.paraphrased_questions:
        for config in configs:
            messages = AskParaphrasedQuestionFormatter.format_example(
                paraphrased_question=paraphrased_question.paraphrased
            )
            ts = ParaphrasedTaskSpec(
                messages=messages,
                formatter_name=AskParaphrasedQuestionFormatter.name(),
                data_example=x.task_spec.data_example,
                inference_config=config,
                task_name=x.task_spec.task_name,
                paraphrased_question=paraphrased_question,
            )
            specs.append(ts)
    return specs


EXP_DIR = "experiments/automated_prompt_variant_generation/v1"


async def run_pipeline(
    exp_dir: str = EXP_DIR,
    example_cap: int = 200,
    tasks: Sequence[str] = COT_TESTING_TASKS,
    batch_size: int = 50,
    eval_temp: float = 0.0,
    models_to_evaluate: Sequence[str] = [],
    paraphrasing_formatters: Sequence[Type[StageOneFormatter]] = [GenerateParaphrasingsFormatters],
) -> Path:
    cache_dir = f"{exp_dir}/cache"

    generation_caller = UniversalCaller().with_file_cache(f"{cache_dir}/generation_cache.jsonl", write_every_n=200)

    answer_parsing_caller = UniversalCaller().with_model_specific_file_cache(
        f"{cache_dir}/answer_parsing_cache", write_every_n=200
    )
    answer_parsing_config = config_from_default(model="claude-2")

    data_examples = get_examples_for_tasks(tasks, example_cap)
    n_items = len(data_examples)

    pipeline = (
        Observable.from_iterable(data_examples)
        .map(
            lambda x: data_to_task_spec(
                *x,
                formatters=paraphrasing_formatters,
                models=[config_from_default(model="gpt-4", max_tokens=3000)],
            )
        )
        .flatten_iterable()
        .map_blocking_par(lambda x: call_model_with_task_spec(x, generation_caller), max_par=batch_size)
        .flatten_list()
        .tqdm(tqdm_bar=tqdm(total=n_items * len(paraphrasing_formatters), desc="Generating prompts"))
        .map(parse_responses)
    )
    if models_to_evaluate:
        models_to_be_tested = Slist(models_to_evaluate).map(
            lambda x: config_from_default(model=x, temperature=eval_temp)
        )
        testing_caller = UniversalCaller().with_model_specific_file_cache(
            f"{cache_dir}/evaluation_cache", write_every_n=200
        )

        pipeline = (
            pipeline.map(lambda x: reformulate_questions_for_asking(x, models_to_be_tested))
            .flatten_iterable()
            .map_blocking_par(lambda x: call_model_with_task_spec(x, testing_caller), max_par=batch_size)
            .tqdm(tqdm_bar=tqdm(total=n_items * 10 * len(models_to_evaluate), desc="Asking parahrased questions"))
            .flatten_list()
            .map_blocking_par(
                lambda x: answer_finding_step(x, answer_parsing_caller, answer_parsing_config), max_par=10
            )
            .tqdm(tqdm_bar=tqdm(total=n_items * 10 * len(models_to_evaluate), desc="Evaluating models"))
        )

    results_dir = Path(f"{exp_dir}/results")
    results = await pipeline.to_slist()
    save_per_model_results(results, results_dir)

    return results_dir


def make_training_data(
    exp_dir="experiments/automated_prompt_variant_generation/training_data",
    example_cap: int = 2500,
    tasks: Sequence[str] = COT_TRAINING_TASKS,
    paraphrasing_formatters: Sequence[Type[StageOneFormatter]] = [
        GenerateParaphrasingsFormatters,
        GenerateParaphrasingsNoCotFormatters,
    ],
    gold_standard_formatters: Sequence[Type[StageOneFormatter]] = [
        GoldStandardWithCotFormatter,
        GoldStandardNoCotFormatter,
        GoldStandardWithCotFormatter2,
        GoldStandardWithCotFormatter3,
        GoldStandardWithCotFormatter4,
    ],
    batch_size=50,
    num_completions_per_prompt: int = 1,
):
    # This generates the different paraphrasings of the questions
    results_path = asyncio.run(
        run_pipeline(
            exp_dir=exp_dir,
            example_cap=example_cap,
            tasks=tasks,
            batch_size=batch_size,
            eval_temp=0.0,
            paraphrasing_formatters=paraphrasing_formatters,
        )
    )
    paraphrased_questions = load_per_model_results(results_path, ParaphrasingOutput)
    write_jsonl_file_from_basemodel(Path("data/training_paraphrasings/gpt4_paraphrasings.jsonl"), paraphrased_questions)

    # but we also want to generate the gold standard completions that we will use to train the model
    # dont need to use any paraphrasings here

    async def get_gold_standard_cots(
        exp_dir: str,
        example_cap: int,
        tasks: Sequence[str],
        batch_size: int,
        num_completions_per_prompt: int = 1,
    ):
        model_caller = UniversalCaller().with_file_cache(
            f"{exp_dir}/cache/cot_generation_cache.jsonl", write_every_n=20
        )
        data_examples = get_examples_for_tasks(tasks, example_cap)
        pipeline = (
            Observable.from_iterable(data_examples)
            .map(
                lambda x: data_to_task_spec(
                    *x,
                    formatters=gold_standard_formatters,
                    models=[config_from_default(model="gpt-3.5-turbo", max_tokens=3000, n=num_completions_per_prompt)],
                )
            )
            .flatten_iterable()
            .map_blocking_par(lambda x: call_model_with_task_spec(x, model_caller), max_par=batch_size)
            .flatten_iterable()
            .tqdm(
                tqdm_bar=tqdm(
                    total=len(data_examples) * len(gold_standard_formatters),
                    desc="Generating Gold Standard Completions",
                )
            )
        )

        results_path = Path(f"{exp_dir}/gold_standard_completions.jsonl")
        # delete the file if it exists
        if results_path.exists():
            results_path.unlink()
        await pipeline.to_file(results_path, mode="a", serialize=lambda x: x.model_dump_json())

        outputs = read_jsonl_file_into_basemodel(results_path, StreamingTaskOutput)

        # Save the outputs to different files based on the formatter
        groupby = outputs.group_by(lambda x: x.task_spec.formatter_name)
        groupby.for_each(lambda x: write_jsonl_file_from_basemodel(Path(f"data/training_cots/{x.key}.jsonl"), x.values))

    asyncio.run(
        get_gold_standard_cots(
            exp_dir=exp_dir,
            example_cap=example_cap,
            tasks=tasks,
            batch_size=batch_size,
            num_completions_per_prompt=num_completions_per_prompt,
        )
    )


class Entropy(BaseModel):
    entropy: float
    uniform_entropy: float


def entropy_and_uniform_entropy(outputs: Sequence[StreamingTaskOutput]) -> Entropy:
    inference_outputs = Slist(outputs).map(lambda x: x.inference_output)
    parsed_responses = inference_outputs.map(lambda x: x.parsed_response)
    counts = Counter(parsed_responses)
    print(counts)
    probabilities = {k: v / len(outputs) for k, v in counts.items()}
    entropy = -sum([p * math.log(p, 2) for p in probabilities.values()])

    # also return the entropy as if the model was uniform
    # get the number of options from the question
    num_options = Slist(outputs).map(lambda x: x.task_spec.n_options_given)
    assert len(num_options.distinct()) == 1
    uniform_prob = 1 / num_options[0]
    uniform_entropy = -sum([uniform_prob * math.log(uniform_prob, 2) for _ in range(num_options[0])])
    return Entropy(entropy=entropy, uniform_entropy=uniform_entropy)


grouped_outputs = Group[tuple[str, OpenaiInferenceConfig], Entropy]


class Extractor(BaseExtractor[grouped_outputs]):
    column_names: list[str] = ["task_hash", "model", "model_with_temp", "temperature", "entropy", "uniform_entropy"]

    def extract(self, output: grouped_outputs) -> Sequence[str | float | None | bool]:
        task_hash = output[0][0]
        model = output[0][1].model
        temperature = output[0][1].temperature
        entropy = output[1]
        return [task_hash, model, f"{model}, t={temperature}", temperature, entropy.entropy, entropy.uniform_entropy]


def plot(exp_dir="experiments/automated_prompt_variant_generation/v1"):
    results_dir = f"{exp_dir}/results"
    outputs = load_per_model_results(Path(results_dir), StreamingTaskOutput)

    # calculate the entropy
    with_entropy = outputs.group_by(lambda x: (x.task_spec.get_task_hash(), x.task_spec.inference_config)).map(
        lambda x: x.map_values(entropy_and_uniform_entropy)
    )

    df = convert_slist_to_df(with_entropy, [Extractor()])

    avg_entropy = df.uniform_entropy.mean()
    catplot(data=df, x="model_with_temp", y="entropy", add_line_at=avg_entropy)

    plt.show()


def lineplot_util(df_p: pd.DataFrame, title: str):
    avg_entropy = df_p.uniform_entropy.mean()
    print("avg entropy", avg_entropy)
    _, ax = plt.subplots(figsize=(6, 6))
    ax = sns.lineplot(
        df_p,
        x="Samples",
        y="entropy",
        hue="Trained on COTS from",
        err_style="bars",
        ax=ax,
    )
    ax.axhline(avg_entropy, ls="--", color="red")
    ax.set_ylabel("Entropy")
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_ylim(0, None)
    # set legend below plot
    # ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    plt.tight_layout()


SWEEPS_DB = SweepDatabase()
SWEEPS_DB.add(Sweeps.zero_shot)
SWEEPS_DB.add(Sweeps.few_shot)


def plot_scaling_curves(
    exp_dir=EXP_DIR,
    model_meta: Sequence[ModelTrainMeta] = SWEEPS_DB.all_models,
):
    model_name_to_meta = Slist(model_meta).map(lambda x: (x.name, x)).to_dict()
    models = Slist(model_meta).map(lambda x: x.name)
    # results_dir = f"{exp_dir}/results"

    results_dir = asyncio.run(
        run_pipeline(
            exp_dir=exp_dir,
            models_to_evaluate=models,
            tasks=COT_TESTING_TASKS,
            batch_size=20,
            eval_temp=0.0,
        )
    )

    outputs = load_per_model_results(results_dir, StreamingTaskOutput)
    with_entropy = outputs.group_by(lambda x: (x.task_spec.get_task_hash(), x.task_spec.inference_config)).map(
        lambda x: x.map_values(entropy_and_uniform_entropy)
    )

    df = convert_slist_to_df(with_entropy, [Extractor()])
    df["Trained on COTS from"] = df.model.map(lambda x: model_name_to_meta[x].for_legend())
    df["Samples"] = df.model.map(lambda x: model_name_to_meta[x].trained_samples)
    df = add_point_at_1(df, "gpt-3.5-turbo")

    lineplot_util(df, title="Entropy across paraphrased questions")

    plt.show()


if __name__ == "__main__":
    fire.Fire(
        {
            "run": run_pipeline,
            "plot": plot,
            "make_training_data": make_training_data,
            "plot_scaling_curves": plot_scaling_curves,
        }
    )
