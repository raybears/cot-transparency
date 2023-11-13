import asyncio
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
from cot_transparency.data_models.config import config_from_default
from cot_transparency.data_models.data import COT_TESTING_TASKS
from cot_transparency.data_models.models import BaseTaskOutput

from cot_transparency.data_models.pd_utils import (
    BaseExtractor,
    BasicExtractor,
    BiasExtractor,
    convert_slist_to_df,
)
from cot_transparency.data_models.streaming import StreamingTaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.name_mapping import name_to_formatter
from cot_transparency.streaming.tasks import (
    call_model_with_task_spec,
    data_to_task_spec,
    get_examples_for_tasks,
)
from scripts.automated_answer_parsing.answer_parsing_example import answer_finding_step
from scripts.prompt_sen_bias_generalization.model_sweeps.paraphrasing import (
    BASELINE_1_W_VERBALIZE,
    PARAPHRASING_1_GS2_UNFILTERED,
    PARAPHRASING_1_GS3_UNFILTERED,
    PARAPHRASING_1_GS4_UNFILTERED,
    PARAPHRASING_1_GS_UNFILTERED,
    PARAPHRASING_1_W_VERBALIZE,
)
from scripts.prompt_sen_bias_generalization.util import (
    add_point_at_1,
    lineplot_util,
    load_per_model_results,
    save_per_model_results,
)
from scripts.prompt_sen_bias_generalization.model_sweeps import (
    SweepDatabase,
    Sweeps,
)
from scripts.prompt_sen_experiments.hand_written.bias_eval import (
    AverageOptionsExtractor,
    BiasTypeExtractor,
)
from scripts.simple_formatter_names import FORMATTER_TO_SIMPLE_NAME
from scripts.training_formatters import TRAINING_COT_FORMATTERS

TEST_FORMATTERS = [f for f in TRAINING_COT_FORMATTERS]

SWEEPS_DB = SweepDatabase()
# SWEEPS_DB.add(Sweeps.paraphrasing_1)
# SWEEPS_DB.add(Sweeps.paraphrasing_2)
# SWEEPS_DB.add(Sweeps.gs_unbiased)


# Previously presented comparison
# SWEEPS_DB.add(Sweeps.gpt)
# SWEEPS_DB.add(Sweeps.prompt_variants_rand)
# SWEEPS_DB.add(Sweeps.zero_shot)
# SWEEPS_DB.add(Sweeps.few_shot)
# SWEEPS_DB.add(Sweeps.og_control)

# Everything using 2
# SWEEPS_DB.add(Sweeps.gpt)
# # SWEEPS_DB.add(Sweeps.prompt_variants_2)
# # # SWEEPS_DB.add(Sweeps.paraphrasing_2_correct)
# SWEEPS_DB.add(Sweeps.paraphrasing_2)
# SWEEPS_DB.add(Sweeps.zero_shot_2)
# SWEEPS_DB.add(Sweeps.few_shot_2)
# SWEEPS_DB.add(Sweeps.og_control)

# SWEEPS_DB = SweepDatabase()
# SWEEPS_DB.add(Sweeps.zero_shot)
# SWEEPS_DB.add(Sweeps.few_shot)

# SWEEPS_DB.add(Sweeps.zero_shot_2)
# SWEEPS_DB.add(Sweeps.few_shot_2)
# SWEEPS_DB.add(Sweeps.paraphrasing_2_correct)
# SWEEPS_DB.add(Sweeps.paraphrasing_2_ba)
# SWEEPS_DB.add(Sweeps.paraphrasing_2)
# SWEEPS_DB.add(Sweeps.prompt_variants_2)
# SWEEPS_DB.add(Sweeps.paraphrasing_2_few_2)
# SWEEPS_DB.add(Sweeps.paraphrasing_2_zero_2)


# SWEEPS_DB.add(Sweeps.zero_shot)
# SWEEPS_DB.add(Sweeps.few_shot)
# SWEEPS_DB.add(Sweeps.zero_shot_2)
# SWEEPS_DB.add(Sweeps.few_shot_2)

# SWEEPS_DB.add(PARAPHRASING_1_GS_UNFILTERED)
# SWEEPS_DB.add(PARAPHRASING_1_GS2_UNFILTERED)
# SWEEPS_DB.add(PARAPHRASING_1_GS3_UNFILTERED)
# SWEEPS_DB.add(PARAPHRASING_1_GS4_UNFILTERED)

SWEEPS_DB.add(Sweeps.gpt)
# SWEEPS_DB.add(Sweeps.zero_shot_2)
# SWEEPS_DB.add(Sweeps.few_shot_2)
# SWEEPS_DB.add(Sweeps.paraphrasing_2_correct)
# SWEEPS_DB.add(Sweeps.paraphrasing_2_ba)
SWEEPS_DB.add(PARAPHRASING_1_W_VERBALIZE)
SWEEPS_DB.add(BASELINE_1_W_VERBALIZE)


async def run_bias_eval(
    exp_dir: str = "experiments/finetune_3_streaming_cc",
    tasks: Sequence[str] = COT_TESTING_TASKS,
    biases: Sequence[Type[StageOneFormatter]] = TEST_FORMATTERS,
    example_cap: int = 400,
    batch: int = 50,
    model_names: Sequence[str] = SWEEPS_DB.all_model_names,
) -> Slist[StreamingTaskOutput]:
    cache_dir = f"{exp_dir}/cache"

    print(f"Running with {len(model_names)} models")
    print(f"Running with {len(biases)} formatters")
    print(f"Running with {len(tasks)} tasks")
    configs = Slist(model_names).map(lambda x: config_from_default(model=x))

    model_caller = UniversalCaller().with_model_specific_file_cache(
        f"{cache_dir}/evaluation_cache",
        write_every_n=200,
    )
    answer_parsing_caller = UniversalCaller().with_model_specific_file_cache(
        f"{cache_dir}/answer_parsing_cache",
        write_every_n=200,
    )
    answer_parsing_config = config_from_default(model="claude-2")

    data = get_examples_for_tasks(tasks, example_cap=example_cap)
    tasks_to_run = data.map(lambda x: data_to_task_spec(x[0], x[1], biases, configs)).flatten_list()

    obs = (
        Observable.from_iterable(tasks_to_run)
        .map_blocking_par(
            lambda x: call_model_with_task_spec(x, model_caller),
            max_par=batch,
        )
        .flatten_list()
        .tqdm(tqdm_bar=tqdm(total=len(tasks_to_run), desc="Evaluate models"))
        .map_blocking_par(lambda x: answer_finding_step(x, answer_parsing_caller, answer_parsing_config), max_par=10)
        .tqdm(tqdm_bar=tqdm(total=len(tasks_to_run), desc="Parsing answers with claude"))
    )

    results = await obs.to_slist()
    results_dir = Path(f"{exp_dir}/results")
    save_per_model_results(results, results_dir)
    return results


class IsCorrectExtractor(BaseExtractor[BaseTaskOutput]):
    column_names = ["is_correct"]

    def extract(self, output: BaseTaskOutput) -> Sequence[str | float | None]:
        return [output.get_task_spec().get_data_example_obj().ground_truth == output.inference_output.parsed_response]


def plot(
    results_dir: str = "experiments/finetune_3_streaming_cc/results",
    tasks: Sequence[str] = COT_TESTING_TASKS,
    biases: Sequence[Type[StageOneFormatter]] = TEST_FORMATTERS,
    plot_breakdown_by_formatter: bool = False,
    example_cap: int = 200,
    force_rerun: bool = False,
):
    sweep_database = SWEEPS_DB

    defined_meta = sweep_database.all_models
    models = sweep_database.all_model_names

    outputs = load_per_model_results(results_dir, StreamingTaskOutput, model_names=models)
    loaded_models = outputs.map(lambda x: x.get_task_spec().inference_config.model).distinct()
    rerun = False
    for model in models:
        if model not in loaded_models:
            rerun = True
            print(f"Didn't find all models requested in {results_dir}. Running evaluation again.")
            break

    if rerun or force_rerun:
        outputs = asyncio.run(run_bias_eval(model_names=sweep_database.all_model_names, example_cap=example_cap))

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
            IsCorrectExtractor(),
        ],
    )
    df["matches_bias"] = df.bias_ans == df.parsed_response

    model_name_to_meta = defined_meta.map(lambda x: (x.name, x)).to_dict()

    df["Trained on COTS from"] = df.model.map(lambda x: model_name_to_meta[x].for_legend())
    df["Samples"] = df.model.map(lambda x: model_name_to_meta[x].trained_samples)

    df = add_point_at_1(df, "gpt-3.5-turbo")

    # drop any sets of "Trained on COTS from" that only have samples at 1
    df = df.groupby("Trained on COTS from").filter(lambda x: x.Samples.max() > 1)  # type: ignore
    df["Accuracy"] = df["is_correct"]  # type: ignore
    df["IsNone"] = 1 * df["parsed_response"].isna()

    # get columns of counts
    counts_df = df.groupby(["Trained on COTS from", "bias_type", "Samples"]).agg({"bias_ans": "count"}).reset_index()

    for bias_type in df.bias_type.unique():
        df_p = df[df.bias_type == bias_type]
        counts_df_p = counts_df[counts_df.bias_type == bias_type]
        title = "Bias type: " + bias_type
        assert isinstance(df_p, pd.DataFrame)
        chance_response = 1 / df_p.average_options.mean()
        lineplot_util(df_p, title, add_line_at=chance_response, y="matches_bias")
        lineplot_util(
            df_p,
            title + " Accuracy",
            y="Accuracy",
        )
        lineplot_util(df_p, title + " Proportion None", y="IsNone")
        lineplot_util(counts_df_p, title + " Counts", y="bias_ans")
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
    fire.Fire({"plot": plot, "run": run_bias_eval})
