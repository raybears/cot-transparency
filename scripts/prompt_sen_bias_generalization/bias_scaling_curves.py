from pathlib import Path
from typing import Type

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from git import Sequence
from slist import Slist

from cot_transparency.data_models.io import read_all_for_selections
from cot_transparency.data_models.pd_utils import (
    BasicExtractor,
    BiasExtractor,
    convert_slist_to_df,
)
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.name_mapping import name_to_formatter
from scripts.finetune_zero_shot_experiments.comparison_plot import ModelTrainMeta
from scripts.prompt_sen_bias_generalization.combinations import (
    bias_vs_prompts,
    n_formats,
    n_questions_comparison,
)
from scripts.prompt_sen_experiments.hand_written.bias_eval import (
    AverageOptionsExtractor,
    BiasTypeExtractor,
)
from scripts.simple_formatter_names import FORMATTER_TO_SIMPLE_NAME
from scripts.training_formatters import TRAINING_COT_FORMATTERS
from stage_one import COT_TESTING_TASKS, main as stage_one_main

TEST_FORMATTERS = [f for f in TRAINING_COT_FORMATTERS]


def get_name_of_run(i: ModelTrainMeta) -> str:
    return (
        f"{i.train_formatters.value}, {i.filter_strategy.value}, {i.sampling_strategy}"
    )


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


def get_models(group_to_plot: str) -> Slist[ModelTrainMeta]:
    if group_to_plot == "n_questions_comparison":
        defined_meta: Slist[ModelTrainMeta] = n_questions_comparison()
    elif group_to_plot == "bias_vs_prompts":
        defined_meta = bias_vs_prompts()
    elif group_to_plot == "n_formats":
        defined_meta = n_formats()
    else:
        raise ValueError(f"Unknown group_to_plot {group_to_plot}")
    return defined_meta


def run(
    group_to_plot: str = "n_questions_comparison",
    exp_dir: str = "experiments/finetune_3",
    tasks: Sequence[str] = COT_TESTING_TASKS,
    biases: Sequence[Type[StageOneFormatter]] = TEST_FORMATTERS,
):
    models = get_models(group_to_plot)
    model_names = models.map(lambda x: x.name)

    stage_one_main(
        exp_dir=exp_dir,
        models=model_names,
        tasks=tasks,
        formatters=Slist(biases).map(lambda x: x.name()),
        example_cap=400,
        raise_after_retries=False,
        temperature=1.0,
        batch=25,
    )


def plot(
    group_to_plot: str = "n_questions_comparison",
    exp_dir: str = "experiments/finetune_3",
    tasks: Sequence[str] = COT_TESTING_TASKS,
    biases: Sequence[Type[StageOneFormatter]] = TEST_FORMATTERS,
    plot_breakdown_by_formatter: bool = False,
):
    defined_meta = get_models(group_to_plot)
    models = [m.name for m in defined_meta]
    outputs = read_all_for_selections(
        exp_dirs=[Path(exp_dir)],
        formatters=[i.name() for i in biases],
        models=models,
        tasks=tasks,
    )

    # convert to dataframe

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

    df["Trained on COTS from"] = df.model.map(
        lambda x: get_name_of_run(model_name_to_meta[x])
    )
    df["Samples"] = df.model.map(lambda x: model_name_to_meta[x].trained_samples)

    baseline = df[df.model == "gpt-3.5-turbo"].copy()
    print("baseline len", len(baseline))
    for meta in defined_meta:
        name_of_run = get_name_of_run(meta)
        # if name of run not in df, duplicate all rows with the same model name and add the name of run
        model = meta.name
        if (
            len(df[(df.model == model) & (df["Trained on COTS from"] == name_of_run)])
            == 0
        ):
            new_rows = baseline.copy()
            new_rows["Trained on COTS from"] = name_of_run
            df = pd.concat((df, new_rows))  # type: ignore

    for bias_type in df.bias_type.unique():
        df_p = df[df.bias_type == bias_type]
        title = "Bias type: " + bias_type
        assert isinstance(df_p, pd.DataFrame)
        # lineplot_util(df_p, title)
        # breakpoint()
        # convert model to str, samples to int and matches bias to float
        df_pt = df_p.copy()
        df_pt["Samples"] = df_pt["Samples"].astype(int)
        df_pt["matches_bias"] = df_pt["matches_bias"].astype(float)
        df_pt["model_name"] = df_pt["Trained on COTS from"].astype(str)
        g = sns.barplot(data=df_pt, x="Samples", hue="model_name", y="matches_bias")
        g.set_title(title)
        plt.show()

    if plot_breakdown_by_formatter:
        for formatter_name in df.formatter_name.unique():
            df_p = df[df.formatter_name == formatter_name]
            assert isinstance(df_p, pd.DataFrame)
            title = (
                "Formatter Name: "
                + FORMATTER_TO_SIMPLE_NAME[name_to_formatter(formatter_name)]
                if name_to_formatter(formatter_name) in FORMATTER_TO_SIMPLE_NAME
                else formatter_name
            )
            lineplot_util(df_p, title)

    plt.show()


if __name__ == "__main__":
    fire.Fire({"run": run, "plot": plot})
