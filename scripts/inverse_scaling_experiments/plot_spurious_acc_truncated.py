import asyncio
from pathlib import Path
from grugstream import Observable

from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.data import InverseScalingTask
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.unbiased import (
    ZeroShotUnbiasedFormatter,
)
from cot_transparency.formatters.inverse_scaling.no_few_shot import (
    InverseScalingOneShotNoCOT,
    RemoveInverseScalingFewShotsNoCOT,
)
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.ignored_reasoning.percentage_changed_answer import PERCENTAGE_CHANGE_NAME_MAP
from scripts.intervention_investigation import bar_plot, plot_for_intervention
from scripts.multi_accuracy import PlotInfo


def formatter_to_plot_name(formatter: type[StageOneFormatter]) -> str:
    if formatter is RemoveInverseScalingFewShotsNoCOT:
        return "Zero shot"
    if formatter is InverseScalingOneShotNoCOT:
        return "One shot"
    if formatter is ZeroShotUnbiasedFormatter:
        return "Original spurious few shot"
    else:
        raise ValueError(f"Unknown formatter {formatter}")


async def plot_accuracies():
    # model = "gpt-3.5-turbo-0613"
    model = "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MmNKzZh"  # ours (superdataset, without few shot)
    # model =  "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MK49rPG"  # control for superdataset
    stage_one_path = Path("experiments/inverse_scaling/stage_one.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=1_000)
    # task = InverseScalingTask.repetitive_algebra
    # task = InverseScalingTask.memo_trap
    # ZeroShotCOTUnbiasedFormatter
    # ZeroShotCOTUnbiasedRepeatMistakesFormatter
    formatters = [RemoveInverseScalingFewShotsNoCOT, InverseScalingOneShotNoCOT, ZeroShotUnbiasedFormatter]
    stage_one_obs: Observable[TaskOutput] = stage_one_stream(
        formatters=[formatter.name() for formatter in formatters],
        tasks=[InverseScalingTask.hindsight_neglect],
        # sample 10 times because hindsight neglect doesn't have many samples
        # we want something similar to "loss" but don't have access to log probs
        example_cap=300,
        n_responses_per_request=10,
        num_tries=1,
        raise_after_retries=False,
        temperature=1.0,
        caller=stage_one_caller,
        batch=40,
        models=[model],
    )

    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    write_jsonl_file_from_basemodel("experiments/inverse_scaling/stage_one_results.jsonl", results)
    results_filtered = results.filter(lambda x: x.first_parsed_response is not None)

    stage_one_caller.save_cache()

    plot_dots: list[PlotInfo] = [
        plot_for_intervention(
            results_filtered,
            intervention=None,
            for_formatters=[f],
            model=model,
            name_override=formatter_to_plot_name(f),
            distinct_qns=False,
        )
        for f in formatters
    ]

    name_override_plotly = PERCENTAGE_CHANGE_NAME_MAP.copy()
    # change \n to <br> for plotly
    for key, value in name_override_plotly.items():
        name_override_plotly[key] = value.replace("\n", "<br>")
    # task_nice_format = task.replace("_", " ").title()
    bar_plot(
        plot_infos=plot_dots,
        title="Accuracy on spurious few shot tasks in Inverse Scaling dataset<br>",
        dotted_line=None,
        y_axis_title="Accuracy",
        name_override=name_override_plotly,
        add_n_to_name=True,
        max_y=1.0,
    )


if __name__ == "__main__":
    asyncio.run(plot_accuracies())
