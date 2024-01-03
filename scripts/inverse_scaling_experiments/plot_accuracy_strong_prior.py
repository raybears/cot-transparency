import asyncio
from pathlib import Path

from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.config import config_from_default
from cot_transparency.data_models.data import InverseScalingTask
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.inverse_scaling.repeat_mistakes import (
    ZeroShotCOTUnbiasedFollowInstructionsFormatter,
)
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.automated_answer_parsing.answer_parsing_example import answer_finding_step
from scripts.intervention_investigation import plot_for_intervention


from enum import Enum
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic import BaseModel

from scripts.multi_accuracy import AccuracyOutput, PlotInfo


class Method(str, Enum):
    ours = "Anti Bias Training"
    control = "Self Traning (Control)"


class ModelTrainMeta(BaseModel):
    name: str
    proportion_instruct_data: float
    trained_on: Method


class ModelNameAndTrainedSamplesAndMetrics(BaseModel):
    train_meta: ModelTrainMeta
    metrics: AccuracyOutput


def read_metric_from_meta(
    meta: ModelTrainMeta,
    all_tasks: Slist[TaskOutput],
) -> ModelNameAndTrainedSamplesAndMetrics:
    # read the metric from the meta
    all_tasks = all_tasks.filter(lambda x: x.task_spec.inference_config.model == meta.name)

    acc: PlotInfo = plot_for_intervention(all_tasks=all_tasks)
    return ModelNameAndTrainedSamplesAndMetrics(train_meta=meta, metrics=acc.acc)


def samples_meta() -> Slist[ModelTrainMeta]:
    # fill this up from wandb https://wandb.ai/raybears/consistency-training?workspace=user-chuajamessh
    """
    _100k_0_perc_new="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8aauYoO9",
    _100k_1_perc_new="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8aanfHwN",
    _100k_5_perc_new="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8aj7xOvu",
    _100k_10_perc_new="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8ab2bFlv",
    _100k_25_perc_new="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8aDG4tqK",
    _100k_50_perc_new="ft:gpt-3.5-turbo-0613:academicsnyuperez::8aDRJnSG",
    _100k_100_perc_new="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8aCHrIH2",
    """
    all_meta = Slist(
        [
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8aCHrIH2",
                proportion_instruct_data=0,
                trained_on=Method.ours,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UfaPIrG",
                proportion_instruct_data=0.5,
                trained_on=Method.ours,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8aDG4tqK",
                proportion_instruct_data=0.75,
                trained_on=Method.ours,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8ab2bFlv",
                proportion_instruct_data=0.9,
                trained_on=Method.ours,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8aj7xOvu",
                proportion_instruct_data=0.95,
                trained_on=Method.ours,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8aanfHwN",
                proportion_instruct_data=0.99,
                trained_on=Method.ours,
            ),
        ]
    )
    distinct_models = all_meta.distinct_by(lambda i: i.name)
    assert len(distinct_models) == len(all_meta), "There are duplicate models in the list"
    return distinct_models


def read_all_metrics(
    samples: Slist[ModelTrainMeta],
    all_tasks: Slist[TaskOutput],
) -> Slist[ModelNameAndTrainedSamplesAndMetrics]:
    return samples.map(lambda meta: read_metric_from_meta(meta, all_tasks))


def seaborn_line_plot(
    data: Sequence[ModelNameAndTrainedSamplesAndMetrics],
    error_bars: bool = True,
    gpt_35_accuracy: Optional[float] = None,
):
    y_axis_label = "Accuracy %"
    df = pd.DataFrame(
        [
            {
                "Instruct Tuning Sample Proportion": i.train_meta.proportion_instruct_data,
                y_axis_label: i.metrics.accuracy,
                "Error Bars": i.metrics.error_bars,
                "Training method": i.train_meta.trained_on.value,
            }
            for i in data
        ]
    )
    df[y_axis_label] = df[y_axis_label]

    ax = sns.lineplot(data=df, x="Instruct Tuning Sample Proportion", y=y_axis_label)

    if gpt_35_accuracy is not None:
        # draw a horizontal line at the gpt-3.5 accuracy
        # make sure it appears in the legend
        ax.axhline(y=gpt_35_accuracy, color="red", linestyle="--", label="GPT-3.5 accuracy")
        # Add a legend
        ax.legend()

    if error_bars:
        for name, group in df.groupby("Training method"):
            plt.errorbar(
                group["Instruct Tuning Sample Proportion"],
                group[y_axis_label],
                yerr=group["Error Bars"],
                fmt="none",
                capsize=5,
                ecolor="black",
            )
    plt.xticks([0, 0.5, 0.75, 0.99], ["0%", "50%", "75%", "99%"])

    # rotate xticks slightly
    # plt.xticks(rotation=-15)
    # set x tickss to unique
    # plt.xticks(df["Instruct Tuning Sample Proportion"].unique())
    # replot the xticks with linear_scale_to_string
    # plt.xticks(
    #     df["Instruct Tuning Sample Proportion"].unique(),  # type: ignore
    #     [linear_scale_to_string(i) for i in df["Instruct Tuning Sample Proportion"].unique()],
    # )

    plt.ylim(0, 1)
    # Mutliple y-axis ticks by 100
    # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, ], Slist([0, 20, 40, 60, 80, 100]).map(str))
    # make the x ticks the

    # x-axis log scale
    # plt.xscale("log")
    # title
    plt.title("Accuracy on Strong Prior tasks")
    # plt.show()
    # save as pdf
    plt.savefig("strong_prior_tasks.pdf", bbox_inches="tight")


async def plot_accuracies():
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MK49rPG",  # control for superdataset 10k
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MKt0VnY",  # ours (superdataset)
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MmNKzZh",  # ours (superdataset, without few shot)
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8KreNXFv", # control paraphrasing 10k
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Kb1ayZh" # ours paraphrasing 10k
    models = [m.name for m in samples_meta()] + ["gpt-3.5-turbo-0613"]
    stage_one_path = Path("experiments/inverse_scaling/stage_one.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=1000)
    # task = InverseScalingTask.memo_trap
    # ZeroShotCOTUnbiasedFormatter
    # ZeroShotCOTUnbiasedRepeatMistakesFormatter
    formatters = [ZeroShotCOTUnbiasedFollowInstructionsFormatter]
    stage_one_obs = stage_one_stream(
        formatters=[f.name() for f in formatters],
        tasks=[InverseScalingTask.memo_trap, InverseScalingTask.resisting_correction, InverseScalingTask.redefine],
        example_cap=936,
        num_tries=1,
        raise_after_retries=False,
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        models=models,
        n_responses_per_request=1,
    )

    answer_parsing_caller = UniversalCaller().with_model_specific_file_cache(
        "experiments/inverse_scaling/answer_parsing"
    )
    config = config_from_default(model="gpt-4")
    stage_one_obs = stage_one_obs.map_blocking_par(lambda x: answer_finding_step(x, answer_parsing_caller, config))

    results: Slist[TaskOutput] = await stage_one_obs.to_slist()

    ori_gpt_35, other_model_results = results.split_by(
        lambda x: x.task_spec.inference_config.model == "gpt-3.5-turbo-0613"
    )

    # grouped_by_model_and_formatter = results.group_by(
    #     lambda x: (x.task_spec.inference_config.model, x.task_spec.formatter_name)
    # )
    # counts = grouped_by_model_and_formatter.map(
    #     lambda group: group.map_values(lambda x: x.map(lambda val: val.inference_output.parsed_response is None).sum())
    # ).to_dict()

    # for k, v in counts.items():
    #     print(k, v)

    write_jsonl_file_from_basemodel("experiments/inverse_scaling/instruction_following.jsonl", results)

    # Filter out nones
    # results = results.filter(lambda x: x.first_parsed_response is not None)
    stage_one_caller.save_cache()

    defined_meta = samples_meta()
    read_metrics = read_all_metrics(samples=defined_meta, all_tasks=other_model_results)
    # get accuracy for gpt-3.5-turbo-0613
    ori_gpt_35_acc: PlotInfo = plot_for_intervention(ori_gpt_35)

    seaborn_line_plot(read_metrics, gpt_35_accuracy=ori_gpt_35_acc.acc.accuracy)


if __name__ == "__main__":
    asyncio.run(plot_accuracies())
