from collections.abc import Sequence
from pathlib import Path
from typing import Type

import fire
from matplotlib import pyplot as plt

from cot_transparency.data_models.io import read_all_for_selections
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.data_models.pd_utils import (
    BaseExtractor,
    BasicExtractor,
    BiasExtractor,
    convert_slist_to_df,
)
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.name_mapping import name_to_formatter
from scripts.training_formatters import (
    TRAINING_COT_FORMATTERS,
    TRAINING_COT_FORMATTERS_FEW_SHOT,
    TRAINING_COT_FORMATTERS_ZERO_SHOT,
    TRAINING_NO_COT_FORMATTERS_FEW_SHOT,
    TRAINING_NO_COT_FORMATTERS_ZERO_SHOT,
)
from scripts.utils.plots import catplot
from scripts.utils.simple_model_names import MODEL_SIMPLE_NAMES
from stage_one import TASK_LIST, main

MODELS = [
    "gpt-3.5-turbo",
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV",  # James 50/50 model
    # "ft:gpt-3.5-turbo-0613:far-ai::88dVFSpt",  # consistency training guy
    # "ft:gpt-3.5-turbo-0613:far-ai::89d1Jn8z",  # 100
    # "ft:gpt-3.5-turbo-0613:far-ai::89dSzlfs",  # 1000
    # "ft:gpt-3.5-turbo-0613:far-ai::89dxzRjA",  # 10000
    # "ft:gpt-3.5-turbo-0613:far-ai::89figOP6",  # 50000
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::88h1pB4E",  # 50 / 50 unbiased
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::89nGUACf",  # 10k James model trained on few shot left out zero shot
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8B24hv5w",  # 10k, 100% CoT, Few Shot Biases, 0% Instruction
    # # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8B4LIFB7",  # 10k, 100% CoT, Control, 0% Instruction
    # "ft:gpt-3.5-turbo-0613:far-ai::8AhgtHQw",  # 10k, include all formatters
    # "ft:gpt-3.5-turbo-0613:far-ai::8Aic3f0n",  # 50k include all formatters
    # "ft:gpt-3.5-turbo-0613:far-ai::8Ahe3cBv",  # 10k, don't include all formatters
    # "ft:gpt-3.5-turbo-0613:far-ai::8AjeHchR",  # 50k don't include all formatters
    # "ft:gpt-3.5-turbo-0613:far-ai::8BRpCYNt",  # 110, train on zero shot
    # "ft:gpt-3.5-turbo-0613:far-ai::8BSJekFR",  # 1100, train on zero shot
    # "ft:gpt-3.5-turbo-0613:far-ai::8BSeBItZ",  # 11000, train on zero shot
    # "ft:gpt-3.5-turbo-0613:far-ai::8BSkM7rh",  # 22000, train on zero shot
    # "ft:gpt-3.5-turbo-0613:far-ai::8Bk36Zdf",  # 110, train on prompt variants
    # "ft:gpt-3.5-turbo-0613:far-ai::8Bmh8wJf",  # 1100, train on prompt variants
    # "ft:gpt-3.5-turbo-0613:far-ai::8Bn9DgF7",  # 11000, train on prompt variants
    # "ft:gpt-3.5-turbo-0613:far-ai::8Boiwe8c",  # 22000, train on prompt variants
    # Trained on mon 23
    # "ft:gpt-3.5-turbo-0613:far-ai::8CvSFvYq",
    # "ft:gpt-3.5-turbo-0613:far-ai::8CvzjiAL",
    # "ft:gpt-3.5-turbo-0613:far-ai::8CwPS37r",
    # "ft:gpt-3.5-turbo-0613:far-ai::8CwqAHpd",
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Cwy6FOO",
    # "ft:gpt-3.5-turbo-0613:far-ai::8CwyWur7",
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8CxLmnUT"
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8CygHwMu",
    # "ft:gpt-3.5-turbo-0613:far-ai::8Czg32py",
    ## Super dataset scaling
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8CxBtbeH",
    "ft:gpt-3.5-turbo-0613:far-ai::8Czg32py",
    "ft:gpt-3.5-turbo-0613:far-ai::8CwqAHpd",
    "ft:gpt-3.5-turbo-0613:far-ai::8CwFcohP",
    # baselines
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::89NHOL5b",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::89NHOL5b",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::89GzBGx0",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::89LJSEdM",
]

EXP_DIR = "experiments/finetune_3"

# Test on both few shot and zero shot biases
TEST_FORMATTERS = [f.name() for f in TRAINING_COT_FORMATTERS]
DATASET = "cot_testing"


def run():
    main(
        exp_dir=EXP_DIR,
        models=MODELS,
        formatters=TEST_FORMATTERS,
        dataset=DATASET,
        example_cap=400,
        raise_after_retries=False,
        temperature=1.0,
        batch=25,
    )


BIAS_TYPE_MAPPING: dict[str, Sequence[Type[StageOneFormatter]]] = {
    "Zero Shot CoT": TRAINING_COT_FORMATTERS_ZERO_SHOT,
    "Few Shot CoT": TRAINING_COT_FORMATTERS_FEW_SHOT,
    "Zero Shot Non CoT": TRAINING_NO_COT_FORMATTERS_ZERO_SHOT,
    "Few Shot NoN CoT": TRAINING_NO_COT_FORMATTERS_FEW_SHOT,
}


class BiasTypeExtractor(BaseExtractor[TaskOutput]):
    column_names = [
        "bias_type",
    ]

    def extract(self, output: TaskOutput) -> Sequence[str | float | None]:
        formatter = name_to_formatter(output.task_spec.formatter_name)

        for k, v in BIAS_TYPE_MAPPING.items():
            if formatter in v:
                bias_type = k
                break
        else:
            raise ValueError(f"Formatter {formatter} not found in BIAS_TYPE_MAPPING")

        return [
            bias_type,
        ]


class AverageOptionsExtractor(BaseExtractor[TaskOutput]):
    column_names = [
        "average_options",
    ]

    def extract(self, output: TaskOutput) -> Sequence[str | float | None]:
        n_options = len(output.task_spec.get_data_example_obj().get_options())
        return [n_options]


def plot(aggregate_formatters: bool = True):
    # load the data
    tasks = TASK_LIST[DATASET]
    outputs = read_all_for_selections(
        exp_dirs=[Path(EXP_DIR)], formatters=TEST_FORMATTERS, models=MODELS, tasks=tasks
    )
    # filter
    outputs = outputs.filter(
        lambda x: x.task_spec.inference_config.model in MODELS
    ).filter(lambda x: x.task_spec.formatter_name in TEST_FORMATTERS)
    # sort so the order is the same as MODELS
    outputs.sort(key=lambda x: MODELS.index(x.task_spec.inference_config.model))

    # calculate the probability of matching the bias if you answered randomly
    # get number of options for each question

    col = "bias_type"

    # convert to dataframe
    df = convert_slist_to_df(
        outputs, extractors=[BasicExtractor(), BiasExtractor(), BiasTypeExtractor()]
    )
    df["matches_bias"] = df.bias_ans == df.parsed_response

    aggregate_tasks = True
    if aggregate_tasks:
        df["task_name"] = ", ".join([i for i in df.task_name.unique()])  # type: ignore

    df["model"] = df["model"].apply(lambda x: MODEL_SIMPLE_NAMES[x] if x in MODEL_SIMPLE_NAMES else x)  # type: ignore
    df["is_correct"] = df.ground_truth == df.parsed_response

    if aggregate_formatters:
        avg_n_ans = outputs.map(
            lambda x: len(x.task_spec.get_data_example_obj().get_options())
        ).average()
        assert avg_n_ans is not None
        g1 = catplot(
            data=df,
            x="task_name",
            y="matches_bias",
            hue="model",
            kind="bar",
            add_line_at=1 / avg_n_ans,
            col=col,
        )
        g2 = catplot(
            data=df, x="task_name", y="is_correct", hue="model", kind="bar", col=col
        )
    else:
        for formatter in df.formatter_name.unique():
            formatter_df = df[df.formatter_name == formatter]
            avg_n_ans = (
                outputs.filter(lambda x: x.task_spec.formatter_name == formatter)
                .map(lambda x: len(x.task_spec.get_data_example_obj().get_options()))
                .average()
            )
            assert avg_n_ans is not None
            g1 = catplot(
                data=formatter_df,
                x="task_name",
                y="matches_bias",
                hue="model",
                kind="bar",
                add_line_at=1 / avg_n_ans,
                col=col,
            )
            g2 = catplot(
                data=formatter_df,
                x="task_name",
                y="is_correct",
                hue="model",
                kind="bar",
                col=col,
            )
            g1.fig.suptitle(formatter)
            g2.fig.suptitle(formatter)

    plt.show()


if __name__ == "__main__":
    fire.Fire({"run": run, "plot": plot})
    # plot()
