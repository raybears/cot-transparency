import asyncio
from pathlib import Path

from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.data import InverseScalingTask
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.inverse_scaling.repeat_mistakes import (
    ZeroShotCOTUnbiasedFollowInstructionsFormatter,
)
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.ignored_reasoning.percentage_changed_answer import PERCENTAGE_CHANGE_NAME_MAP
from scripts.intervention_investigation import bar_plot, plot_for_intervention
from scripts.multi_accuracy import PlotInfo


async def plot_accuracies():
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MK49rPG",  # control for superdataset 10k
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MKt0VnY",  # ours (superdataset)
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MmNKzZh",  # ours (superdataset, without few shot)
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8KreNXFv", # control paraphrasing 10k
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Kb1ayZh" # ours paraphrasing 10k
    model = "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Kb1ayZh"
    stage_one_path = Path("experiments/inverse_scaling/stage_one.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=1000)
    # task = InverseScalingTask.memo_trap
    # ZeroShotCOTUnbiasedFormatter
    # ZeroShotCOTUnbiasedRepeatMistakesFormatter
    formatters = [ZeroShotCOTUnbiasedFollowInstructionsFormatter]
    stage_one_obs = stage_one_stream(
        formatters=[f.name() for f in formatters],
        tasks=[InverseScalingTask.memo_trap, InverseScalingTask.resisting_correction, InverseScalingTask.redefine],
        example_cap=100,
        num_tries=1,
        raise_after_retries=False,
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        models=["gpt-3.5-turbo-0613", model],
    )

    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    write_jsonl_file_from_basemodel("experiments/inverse_scaling/instruction_following.jsonl", results)
    results_filtered = results.filter(lambda x: x.first_parsed_response is not None)
    stage_one_caller.save_cache()

    plot_dots: list[PlotInfo] = [
        plot_for_intervention(
            results_filtered,
            intervention=None,
            for_formatters=[plot_formatter],
            model=model,
            name_override="Original task"
            if plot_formatter is ZeroShotCOTUnbiasedFormatter
            else "Additional instructions to follow user mistakes",
        )
        for plot_formatter in formatters
    ]

    name_override_plotly = PERCENTAGE_CHANGE_NAME_MAP.copy()
    # change \n to <br> for plotly
    for key, value in name_override_plotly.items():
        name_override_plotly[key] = value.replace("\n", "<br>")
    bar_plot(
        plot_infos=plot_dots,
        title="Accuracy on Strong Prior tasks<brCOT<br>LR=0.4",
        dotted_line=None,
        y_axis_title="Accuracy",
        name_override=name_override_plotly,
        add_n_to_name=True,
        max_y=1.0,
    )


if __name__ == "__main__":
    asyncio.run(plot_accuracies())
