
import asyncio
import json
from pathlib import Path
from typing import Type

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from git import Sequence
from grugstream import Observable
from slist import Slist
from tqdm import tqdm

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.config import config_from_default, OpenaiInferenceConfig
from cot_transparency.data_models.data import COT_TESTING_TASKS

from cot_transparency.data_models.pd_utils import (
    BasicExtractor,
    BiasExtractor,
    convert_slist_to_df,
)
from cot_transparency.formatters.refusal.refusal import RefusalFormatter
from cot_transparency.formatters.name_mapping import name_to_formatter
from cot_transparency.data_models.data.refusal import RefusalExample, load_data
from cot_transparency.streaming.tasks import (
    StreamingTaskOutput,
    StreamingTaskSpec,
    call_model_with_task_spec,
)
from scripts.prompt_sen_bias_generalization.util import (
    add_point_at_1,
    load_per_model_results,
    save_per_model_results,
)
from scripts.redteaming_prompt_sen.model_sweeps import (
    SweepDatabase,
    Sweeps,
)
from scripts.prompt_sen_experiments.hand_written.bias_eval import (
    AverageOptionsExtractor,
    BiasTypeExtractor,
)
from scripts.simple_formatter_names import FORMATTER_TO_SIMPLE_NAME
import asyncio
from typing import Sequence, TypeVar

from grugstream import Observable
from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.base import CachedPerModelCaller
from cot_transparency.data_models.config import OpenaiInferenceConfig, config_from_default
from cot_transparency.data_models.models import BaseTaskOutput
from cot_transparency.formatters.auto_answer_parsing import GetGradeGivenFormatter
from cot_transparency.streaming.tasks import StreamingTaskSpec
from cot_transparency.streaming.tasks import call_model_with_task_spec

A = TypeVar("A", bound=BaseTaskOutput)


def answer_grading_step(prev_output: A, caller: CachedPerModelCaller, config: OpenaiInferenceConfig) -> A:
    """
    For any outputs that were not find in the previous step, pass the raw response to another model model and
    ask it to find the answer in the response.
    """
    # if the previous step did find the answer, then we don't need to do anything
    output = prev_output.inference_output
    model_response = output.raw_response

    # unpack the results from the previous step
    answer_grading_formatter = GetGradeGivenFormatter
    messages = answer_grading_formatter.format_example(
        model_response=model_response,
        original_question=prev_output.get_task_spec().get_data_example_obj(),
        model=config.model,
    )
    task_spec = StreamingTaskSpec(
        messages=messages,
        formatter_name=answer_grading_formatter.name(),
        data_example=prev_output.get_task_spec().get_data_example_obj().model_dump(),
        inference_config=config,
        task_name=prev_output.get_task_spec().get_task_name(),
    )

    # we do this so that we get a seperate cache for each model that generated the answer
    # so we can run this script in parallel without running into cache conflicts between processes
    cache_name = f"{prev_output.get_task_spec().inference_config.model}_{config.model}"
    specific_caller = caller.get_specific_caller(cache_name=cache_name)
    output_of_parsing = call_model_with_task_spec(task_spec, specific_caller)
    assert len(output_of_parsing) == 1, "Expected only one output from the answer parsing model"
    output_of_parsing = output_of_parsing[0]
    found_answer = output_of_parsing.inference_output.parsed_response
    print(f"Response: {model_response}\nOutput: {output_of_parsing.inference_output}\nScore:{found_answer}")

    return prev_output.update_parsed_response(found_answer)


def data_and_jailbreaks_to_task_spec(
    task_name: str,
    x: RefusalExample,
    formatters: Sequence[Type[RefusalFormatter]],
    models: Sequence[OpenaiInferenceConfig],
) -> list[StreamingTaskSpec]:
    specs = []
    jailbreaks_path = f"data/refusal/jailbreaks.json"
    with open(jailbreaks_path, "r") as f:
        jailbreaks = json.load(f)
    for formatter in formatters:
        for model in models:
            for jailbreak in jailbreaks:
                jailbreak_name = jailbreak["name"]
                jailbreak_prompt = jailbreak["prompt"]
                print(jailbreak_prompt)
                x.question = jailbreak_prompt.format(QUESTION=x.question)
                
                messages = formatter.format_example(x)

                ts = StreamingTaskSpec(
                    messages=messages,
                    formatter_name=formatter.name(),
                    data_example=x.model_dump(),
                    inference_config=model,
                    task_name=task_name,
                    jailbreak_name=jailbreak_name,
                    category=x.category,
                )
                specs.append(ts)
    return specs

def lineplot_util(df_p: pd.DataFrame, title: str):
    chance_response = 1 / df_p.average_options.mean()
    _, ax = plt.subplots(figsize=(6, 6))
    ax = sns.lineplot(
        df_p,
        x="Samples",
        y="matches_bias",
        hue="Trained on COTS from",
        err_style="bars",
        ax=ax,
    )
    ax.axhline(chance_response, ls="--", color="red")
    ax.set_ylabel("Proportion of responses matching bias")
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    # set legend below plot
    # ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    plt.tight_layout()


SWEEPS_DB = SweepDatabase()
SWEEPS_DB.add(Sweeps.paraphrasing_1)
# SWEEPS_DB.add(Sweeps.gpt)
# SWEEPS_DB.add(Sweeps.paraphrasing_2)
# SWEEPS_DB.add(Sweeps.prompt_variants_2)
# SWEEPS_DB.add(Sweeps.gs_unbiased)
# SWEEPS_DB.add(Sweeps.zero_shot)
# SWEEPS_DB.add(Sweeps.few_shot)
# SWEEPS_DB.add(Sweeps.og_control)
REFUSAL_FORMATTERS = [RefusalFormatter()]

async def run_refusal_eval(
    exp_dir: str = "experiments/refusal",
    formats: Sequence[Type[RefusalFormatter]] = REFUSAL_FORMATTERS,
    example_cap: int = 400,
    batch: int = 50,
    model_names: Sequence[str] = SWEEPS_DB.all_model_names,
) -> Slist[StreamingTaskOutput]:
    cache_dir = f"{exp_dir}/cache"

    print(f"Running with {len(model_names)} models")
    print(f"Running with {len(formats)} formatters")
    configs = Slist(model_names).map(lambda x: config_from_default(model=x))

    model_caller = UniversalCaller().with_model_specific_file_cache(
        f"{cache_dir}/evaluation_cache",
        write_every_n=200,
    )
    answer_grading_caller = UniversalCaller().with_model_specific_file_cache(
        f"{cache_dir}/answer_grading_cache",
        write_every_n=200,
    )
    answer_grading_config = config_from_default(model="claude-2")
    # answer_grading_config = config_from_default(model="gpt-3.5-turbo")

    data = Slist([("refusal", example) for example in load_data(example_cap)])
    tasks_to_run = data.map(lambda x: data_and_jailbreaks_to_task_spec(x[0], x[1], formats, configs)).flatten_list()

    obs = (
        Observable.from_iterable(tasks_to_run)
        .map_blocking_par(
            lambda x: call_model_with_task_spec(x, model_caller),
            max_par=batch,
        )
        .flatten_list()
        .tqdm(tqdm_bar=tqdm(total=len(tasks_to_run), desc="Evaluate models"))
        .map_blocking_par(lambda x: answer_grading_step(x, answer_grading_caller, answer_grading_config), max_par=batch)
        .tqdm(tqdm_bar=tqdm(total=len(tasks_to_run), desc="Grading answers with claude"))
    )

    results = await obs.to_slist()
    results_dir = Path(f"{exp_dir}/results")
    save_per_model_results(results, results_dir)
    return results


def plot(
    results_dir: str = "experiments/finetune_3_streaming_cc/results",
    tasks: Sequence[str] = COT_TESTING_TASKS,
    biases: Sequence[Type[RefusalFormatter]] = REFUSAL_FORMATTERS,
    plot_breakdown_by_formatter: bool = False,
    example_cap: int = 200,
):
    sweep_database = SWEEPS_DB

    defined_meta = sweep_database.all_models
    models = sweep_database.all_model_names

    outputs = load_per_model_results(results_dir, StreamingTaskOutput)
    loaded_models = outputs.map(lambda x: x.get_task_spec().inference_config.model).distinct()
    for model in models:
        if model not in loaded_models:
            print(f"Didn't find all models requested in {results_dir}. Running evaluation again.")
            outputs = asyncio.run(run_refusal_eval(model_names=sweep_database.all_model_names, example_cap=example_cap))
            break

    outputs = (
        outputs.filter(lambda x: x.task_spec.task_name in tasks)
        .filter(lambda x: x.get_task_spec().inference_config.model in models)
        .filter(lambda x: x.get_task_spec().formatter_name in [f.name() for f in biases])
    )
    print("Num results after filtering", len(outputs))

    df = convert_slist_to_df(
        outputs,
        extractors=[
            BasicExtractor(),
            BiasExtractor(),
            BiasTypeExtractor(),
            AverageOptionsExtractor(),
        ],
    )
    df["matches_bias"] = df.bias_ans == df.parsed_response

    model_name_to_meta = defined_meta.map(lambda x: (x.name, x)).to_dict()

    df["Trained on COTS from"] = df.model.map(lambda x: model_name_to_meta[x].for_legend())
    df["Samples"] = df.model.map(lambda x: model_name_to_meta[x].trained_samples)

    df = add_point_at_1(df, "gpt-3.5-turbo")

    # drop any sets of "Trained on COTS from" that only have samples at 1
    df = df.groupby("Trained on COTS from").filter(lambda x: x.Samples.max() > 1)

    for bias_type in df.bias_type.unique():
        df_p = df[df.bias_type == bias_type]
        title = "Bias type: " + bias_type
        assert isinstance(df_p, pd.DataFrame)
        lineplot_util(df_p, title)
        # breakpoint()
        # convert model to str, samples to int and matches bias to float
        # df_pt = df_p.copy()
        # df_pt["Samples"] = df_pt["Samples"].astype(int)
        # df_pt["matches_bias"] = df_pt["matches_bias"].astype(float)
        # df_pt["model_name"] = df_pt["Trained on COTS from"].astype(str)
        # g = sns.barplot(data=df_pt, x="Samples", hue="model_name", y="matches_bias")
        # g.set_title(title)
        plt.show()

    if plot_breakdown_by_formatter:
        for formatter_name in df.formatter_name.unique():
            df_p = df[df.formatter_name == formatter_name]
            assert isinstance(df_p, pd.DataFrame)
            title = (
                "Formatter Name: " + FORMATTER_TO_SIMPLE_NAME[name_to_formatter(formatter_name)]
                if name_to_formatter(formatter_name) in FORMATTER_TO_SIMPLE_NAME
                else formatter_name
            )
            lineplot_util(df_p, title)

    plt.show()


if __name__ == "__main__":
    fire.Fire({"plot": plot, "run": run_refusal_eval})
