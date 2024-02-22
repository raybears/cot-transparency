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
    fifty_fifty = "50-50"
    no_cot = "No COT"
    cot = "COT"
    gpt_35 = "GPT-3.5"
    control = "Self Trainning (Control)"


class ModelTrainMeta(BaseModel):
    name: str
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
    # b1_intervention="ft:gpt-3.5-turbo-0613:far-ai::8rwdMKOn",
    # b2_intervention="ft:gpt-3.5-turbo-0613:far-ai::8rwNfI72",
    """
    all_meta = Slist(
        [
            ModelTrainMeta(name="gpt-3.5-turbo-0613", trained_on=Method.gpt_35),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8rwdMKOn",
                trained_on=Method.fifty_fifty,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8rwdMKOn",
                trained_on=Method.fifty_fifty,
            ),
            # ModelTrainMeta(trained_on=Method.control, name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UN5nhcE"),
            # # 2k
            # # without few shot ft:gpt-3.5-turbo-0613:academicsnyuperez::8MmNKzZh
            # # all "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MKt0VnY"
            # # ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNz4qzi combined paraphrasing +few shot
            # # all syco variants
            # # ModelTrainMeta(hue="Intervention", model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UNAODuA"),
            # ModelTrainMeta(trained_on=Method.fifty_fifty, name="ft:gpt-3.5-turbo-0613:far-ai::8gArPtjO"),
            # ModelTrainMeta(trained_on=Method.fifty_fifty, name="ft:gpt-3.5-turbo-0613:far-ai::8gAkugeh"),
            # ModelTrainMeta(trained_on=Method.fifty_fifty, name="ft:gpt-3.5-turbo-0613:far-ai::8ZNx8yk5"),
            # ModelTrainMeta(trained_on=Method.no_cot, name="ft:gpt-3.5-turbo-0613:far-ai::8inNukCs"),
            # ModelTrainMeta(trained_on=Method.no_cot, name="ft:gpt-3.5-turbo-0613:far-ai::8inQNPtE"),
            # ModelTrainMeta(trained_on=Method.no_cot, name="ft:gpt-3.5-turbo-0613:far-ai::8iopLeXP"),
            # ModelTrainMeta(trained_on=Method.cot, name="ft:gpt-3.5-turbo-0613:far-ai::8jrpSXpl"),
            # ModelTrainMeta(trained_on=Method.cot, name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8jrsOSGF"),
            # ModelTrainMeta(trained_on=Method.cot, name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8jrfoWFZ"),
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


def seaborn_barplot(
    data: Sequence[ModelNameAndTrainedSamplesAndMetrics],
    error_bars: bool = True,
    gpt_35_accuracy: Optional[float] = None,
):
    y_axis_label = "Accuracy %"
    df = pd.DataFrame(
        [
            {
                y_axis_label: i.metrics.accuracy,
                "Error Bars": i.metrics.error_bars,
                "Training method": i.train_meta.trained_on.value,
            }
            for i in data
        ]
    )

    # Make a bar plot with the hue of the training method
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(
        x="Training method",
        y=y_axis_label,
        data=df,
        palette="Blues_d",
    )
    # write to pdf
    plt.savefig("strong_prior_acc.pdf")
    # show it
    plt.show()




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

    seaborn_barplot(read_metrics, gpt_35_accuracy=ori_gpt_35_acc.acc.accuracy)


if __name__ == "__main__":
    asyncio.run(plot_accuracies())

