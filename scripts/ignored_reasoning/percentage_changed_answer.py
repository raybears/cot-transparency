import asyncio
from pathlib import Path
from typing import Optional, Mapping, Sequence

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from pydantic import BaseModel
from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter, ZeroShotUnbiasedFormatter
from cot_transparency.streaming.stage_one_stream import stage_one_stream


class ChangedAnswer(BaseModel):
    without_cot: TaskOutput
    with_cot: TaskOutput
    changed_answer: bool


def group_into_percentage_changed_answers(results: Slist[TaskOutput]) -> Optional[ChangedAnswer]:
    without_cot_formatter = ZeroShotUnbiasedFormatter
    with_cot_formatter = ZeroShotCOTUnbiasedFormatter
    # assert that there are two results for each group
    assert len(results) == 2
    # get the without_cot_formatter
    without_cot = results.filter(lambda x: x.task_spec.formatter_name == without_cot_formatter.name()).first_or_raise()
    # get the with_cot_formatter
    with_cot = results.filter(lambda x: x.task_spec.formatter_name == with_cot_formatter.name()).first_or_raise()
    without_cot_answer = without_cot.inference_output.parsed_response
    with_cot_answer = with_cot.inference_output.parsed_response
    # if either answer is None, return None
    if without_cot_answer is None or with_cot_answer is None:
        return None
    # check if the answer changed
    changed_answer = without_cot_answer != with_cot_answer
    return ChangedAnswer(without_cot=without_cot, with_cot=with_cot, changed_answer=changed_answer)


def compute_percentage_changed(results: Slist[TaskOutput]) -> Slist[ChangedAnswer]:
    # assert unique model
    distinct_model = results.map(lambda x: x.task_spec.inference_config.model).distinct_item_or_raise(lambda x: x)

    # for each group, check if the answer changed
    computed: Slist[ChangedAnswer] = (
        results.group_by(lambda x: x.task_spec.task_hash)
        .map_2(lambda key, values: group_into_percentage_changed_answers(values))
        .flatten_option()
    )
    print(f"Total number of results for {distinct_model}", len(computed))
    print(f"Average percentage changed {distinct_model}", computed.map(lambda x: x.changed_answer).average_or_raise())
    return computed


class GroupResult(BaseModel):
    model: str
    changed_answers: Slist[ChangedAnswer]


def percentage_changed_per_model(results: Slist[TaskOutput]) -> Slist[GroupResult]:
    # group by model
    grouped = results.group_by(lambda x: x.task_spec.inference_config.model)
    print("Total number of models", len(grouped))
    return grouped.map_2(
        lambda model, values: GroupResult(model=model, changed_answers=compute_percentage_changed(values))
    )


def print_average_length(results: Slist[TaskOutput]) -> None:
    # group by model
    grouped = results.group_by(lambda x: x.task_spec.inference_config.model)
    for model, values in grouped:
        average_length = values.map(lambda x: len(x.inference_output.raw_response)).average_or_raise()
        print(f"Average length for {model} is {average_length}")


def seaborn_bar_plot(results: Slist[GroupResult], name_mapping: Mapping[str, str], order: Sequence[str] = []) -> None:
    _dicts = []
    for group_result in results:
        for changed_answer in group_result.changed_answers:
            _dicts.append(
                {
                    "model": name_mapping.get(group_result.model, group_result.model),
                    "same_answer": not changed_answer.changed_answer,
                }
            )
    order_mapped = [name_mapping.get(model, model) for model in order]
    # x-axis is model
    # y-axis is percentage same
    df = pd.DataFrame(_dicts)
    ax = seaborn.barplot(x="model", y="same_answer", data=df, order=order_mapped)
    # change the y-axis to be "Percentage of questions with same answer with vs without COT"
    ax.set(ylabel="% Same Answer With and Without CoT ")
    plt.show()


def seaborn_bar_plot_length(
    results: Slist[TaskOutput], name_mapping: Mapping[str, str], order: Sequence[str] = []
) -> None:
    only_cot = results.filter(lambda x: x.task_spec.formatter_name == ZeroShotCOTUnbiasedFormatter.name())
    _dicts = []
    for task in only_cot:
        length = len(task.inference_output.raw_response)
        model = task.task_spec.inference_config.model
        _dicts.append(
            {
                "model": name_mapping.get(model, model),
                "COT length": length,
            }
        )
    order_mapped = [name_mapping.get(model, model) for model in order]
    # x-axis is model
    # y-axis is length
    df = pd.DataFrame(_dicts)
    ax = seaborn.barplot(x="model", y="COT length", data=df, estimator=np.average, order=order_mapped)  # type: ignore

    # change the y-axis to be "Median COT length"
    ax.set(ylabel="Median COT length")

    plt.show()


PERCENTAGE_CHANGE_NAME_MAP = {
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez:qma-me-75-25:8AdFi5Hs": "Trained to follow mistakes",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8FenfJNo": "Trained with unbiased contexts (control)\n 98% COT, 100k samples",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8FfN5MGW": "Trained with unbiased contexts (control)\n 98% COT, 10k samples",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8FeFMAOR": "Trained with unbiased contexts (control)\n 98% COT, 1k samples",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8FWFloan": "Trained with biased contexts (ours)\n 98% COT, 100k samples",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Ff8h3yF": "Trained with biased contexts (ours)\n 98% COT, 1k samples",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8FciULKF": "Trained with biased contexts (ours)\n 98% COT, 10k samples",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8FgGQFZg": "Trained with biased contexts (ours)\n 50% COT, 10k samples",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8FgC1oNW": "Trained with biased contexts (ours)\n 2% COT, 10k samples",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8FqqxEJy": "Trained with unbiased contexts (control) \n 98% COT, 10k samples\n correct only",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Fn77EVN": "Trained with biased contexts (ours) \n 98% COT, 10k samples\n correct only",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8FtrLOJx": "Ed's paraphrasing 1k 8FtrLOJx",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8HM5LSlU": "Ed's paraphrasing 10k 8HM5LSlU",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G5HsCmO": "Control gold standard 10k 8G5HsCmO",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G5caiZn": "Ed's paraphrasing 48k 8G5caiZn",
    "ft:gpt-3.5-turbo-0613:far-ai::8J2a3iJg": "lr=0.02, Trained with biased contexts (ours)",  # lr =0.02
    # lr = 0.05
    "ft:gpt-3.5-turbo-0613:far-ai::8J2a3PON": "lr=0.05, Trained with biased contexts (ours)",
    # lr = 0.1
    "ft:gpt-3.5-turbo-0613:far-ai::8J3Z5bnB": "lr=0.1, Trained with biased contexts (ours)",
    # lr = 0.2
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8L931IqD": "100 biased + Instruct 1000%",
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8J4ZG4dt": "lr=0.2, Trained with biased contexts (ours)",
    # lr = 0.4
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8J3nhVak": "lr=0.4, Trained with biased contexts (ours)",
    "ft:gpt-3.5-turbo-0613:far-ai::8JFpXaDd": "1k control + Instruct 10%",  # prop=0.1, control
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8J4ZG4dt": "1k biased + Instruct 10%",  # prop=0.1, ours
    # prop=1.0, control
    "ft:gpt-3.5-turbo-0613:far-ai::8JGAIbOw": "1k control + Instruct 100%",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8JGF6zzt": "1k biased + Instruct 100%",
    # prop=5.0, control
    "ft:gpt-3.5-turbo-0613:far-ai::8JJvJpWl": "1k control + Instruct 500%",
    # prop=5.0, ours
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8JIhHMK1": "1k biased + Instruct 500%",
    # prop=10.0, control
    "ft:gpt-3.5-turbo-0613:far-ai::8JNs7Bf0": "1k control + Instruct 1000%",
    # prop=10.0, ours
    "ft:gpt-3.5-turbo-0613:far-ai::8JMuzOOD": "1k biased + Instruct 1000%",
    # prop=0.1 control
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8JRyoeL1": "1k control + Instruct 10%",
    # prop =0.1, ours
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8JQyTvI4": "1k biased + Instruct 10%",
    # prop=1.0 control
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8JR5a5FJ": "1k control + Instruct 100%",
    # prop=1.0 ours
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8JR64wbx": "1k biased + Instruct 100%",
    # prop=10.0 control
    "ft:gpt-3.5-turbo-0613:far-ai::8KJ85aBY": "1k control + Instruct 1000%",
    "ft:gpt-3.5-turbo-0613:far-ai::8KIPBRpI": "1k biased + Instruct 1000%",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Kxr9E7k": "1k biased + NEW Instruct 10%",
    "ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8L4ATSyN": "1k biased + NEW Instruct 1000%",
    # control 50-50 10k correct
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8JOwa1JV": "10k biased + Instruct 10%",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G14z8Tu": "Trained with unbiased contexts (control)\n 50% COT, 10k samples\n correct only",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G1FW35z": "Trained with biased contexts (ours)\n 50% COT, 10k samples\n correct only",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G28v39j": "Trained with biased contexts (ours)\n 2% COT, 10k samples\n correct only",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G2UrNn0": "Trained with biased contexts (ours)\n 98% COT, 10k samples\n correct only",
    "ft:gpt-3.5-turbo-0613:far-ai::8G1QYyMD": "Trained with biased contexts (ours)\n 2% COT, 10k samples\n unfiltered",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G0tbUsB": "Trained with biased contexts (ours)\n 50% COT, 10k samples\n unfiltered",
    "ft:gpt-3.5-turbo-0613:far-ai::8G1YsULJ": "Trained with biased contexts (ours)\n 98% COT, 10k samples\n unfiltered",
    "ft:gpt-3.5-turbo-0613:far-ai::8G4AfzJQ": "Trained with unbiased contexts (control)\n 98% COT, 10k samples\n correct only",
    "ft:gpt-3.5-turbo-0613:far-ai::8G1NdOHF": "Trained with biased contexts (ours)\n50% COT, 1k samples\n+10% instruct",
    "ft:gpt-3.5-turbo-0613:far-ai::8GQiNe1D": "Trained with unbiased contexts (control)\n50% COT, 1k samples\n+10% instruct",
    "ft:gpt-3.5-turbo-0613:far-ai::8Ho5AmzO": "Trained with unbiased contexts (control)\n50% COT, 1k samples\n+100% instruct",
    "ft:gpt-3.5-turbo-0613:far-ai::8Ho0yXlM": "Trained with biased contexts (ours)\n50% COT, 1k samples\n+100% instruct",
}


async def main():
    stage_one_path = Path("experiments/changed_answer/stage_one.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path)
    # control 1k ft:gpt-3.5-turbo-0613:academicsnyuperez::89G2vwHZ
    # control 100k ft:gpt-3.5-turbo-0613:academicsnyuperez::8A6Ymjb2
    # intervention 1k ft:gpt-3.5-turbo-0613:academicsnyuperez::89hifzfA
    # intervention 100k ft:gpt-3.5-turbo-0613:academicsnyuperez::89ghXobC
    # blessed ft:gpt-3.5-turbo-0613:academicsnyuperez::7yyd6GaT
    # hunar trained ft:gpt-3.5-turbo-0613:academicsnyuperez:qma-me-75-25:8AdFi5Hs
    # super dataset 100k ft:gpt-3.5-turbo-0613:far-ai::8DPAu94W
    models = [
        "gpt-3.5-turbo",
        ### START 2%, 50%, 98% COT UNFILTER
        # "ft:gpt-3.5-turbo-0613:far-ai::8G1QYyMD",  # 2%
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G0tbUsB",  # 50%
        # "ft:gpt-3.5-turbo-0613:far-ai::8G1YsULJ",  # 98%
        ### END
        ### START Hunar's, Control, Ours
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez:qma-me-75-25:8AdFi5Hs",
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G14z8Tu",  # control 50-50 10k correct
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G1FW35z",  # ours 50-50 10k correct
        ## START compare to unfiltered
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G0tbUsB", # ours 50-50 10k unfiltered
        ### START 2%, 50%, 98% COT CORRECT
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G28v39j",  # 2 %
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G1FW35z",  # 50%
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G2UrNn0",  # 98%
        # "ft:gpt-3.5-turbo-0613:far-ai::8G3Avv2Y", # 98% rerun
        "ft:gpt-3.5-turbo-0613:far-ai::8G4AfzJQ",  # control 98% 10k correct
    ]
    stage_one_obs = stage_one_stream(
        formatters=[ZeroShotCOTUnbiasedFormatter.name(), ZeroShotUnbiasedFormatter.name()],
        tasks=["aqua", "mmlu", "truthful_qa", "logiqa"],
        example_cap=600,
        num_tries=1,
        raise_after_retries=False,
        temperature=1.0,
        caller=stage_one_caller,
        batch=40,
        models=models,
    )
    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    stage_one_caller.save_cache()
    percentage_changed = percentage_changed_per_model(results)

    # seaborn_bar_plot_length(
    #     results,
    #     name_mapping=PERCENTAGE_CHANGE_NAME_MAP,
    #     order=models,
    # )
    seaborn_bar_plot(
        percentage_changed,
        name_mapping=PERCENTAGE_CHANGE_NAME_MAP,
        order=models,
    )


if __name__ == "__main__":
    asyncio.run(main())
