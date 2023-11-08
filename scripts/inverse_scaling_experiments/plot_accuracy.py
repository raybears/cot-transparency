import asyncio
from pathlib import Path

from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.data import InverseScalingTask
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.inverse_scaling.repeat_mistakes import AssistantThinksRepeatMistakeFormatter
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.ignored_reasoning.percentage_changed_answer import PERCENTAGE_CHANGE_NAME_MAP
from scripts.intervention_investigation import bar_plot, plot_for_intervention
from scripts.multi_accuracy import PlotInfo


async def plot_accuracies():
    models = [
        "gpt-3.5-turbo",
        "ft:gpt-3.5-turbo-0613:far-ai::8GQiNe1D",  # 1k correct (control)
        "ft:gpt-3.5-turbo-0613:far-ai::8G1NdOHF",  # 1k correct (ours)
        # "ft:gpt-3.5-turbo-0613:far-ai::8HoBQfFE", # 100% instruct 500 (ours)
        "ft:gpt-3.5-turbo-0613:far-ai::8Ho5AmzO",  # 100% instruct 1000, control
        "ft:gpt-3.5-turbo-0613:far-ai::8Ho0yXlM",  # 100% instruct 1000 (ours)
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G1FW35z"
    ]
    stage_one_path = Path("experiments/inverse_scaling/stage_one.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=500)
    task = InverseScalingTask.resisting_correction
    # task = InverseScalingTask.memo_trap
    # ZeroShotCOTUnbiasedFormatter
    stage_one_obs = stage_one_stream(
        formatters=[AssistantThinksRepeatMistakeFormatter.name()],
        tasks=[task],
        example_cap=2000,
        num_tries=1,
        raise_after_retries=False,
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        models=models,
    )

    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    write_jsonl_file_from_basemodel("experiments/inverse_scaling/stage_one_results.jsonl", results)
    results_filtered = results.filter(lambda x: x.first_parsed_response is not None)
    stage_one_caller.save_cache()

    plot_formatter = AssistantThinksRepeatMistakeFormatter

    plot_dots: list[PlotInfo] = [
        plot_for_intervention(
            results_filtered,
            intervention=None,
            for_formatters=[plot_formatter],
            model=model,
            name_override=model,
        )
        for model in models
    ]

    prompt_type_str = "COT prompt" if "COT" in plot_formatter.name() else "Non COT prompt"
    name_override_plotly = PERCENTAGE_CHANGE_NAME_MAP.copy()
    # change \n to <br> for plotly
    for key, value in name_override_plotly.items():
        name_override_plotly[key] = value.replace("\n", "<br>")
    task_nice_format = task.replace("_", " ").title()
    bar_plot(
        plot_infos=plot_dots,
        title=f"Accuracy on {task_nice_format}<br>{prompt_type_str}",
        dotted_line=None,
        y_axis_title="Accuracy",
        name_override=name_override_plotly,
        add_n_to_name=True,
    )


if __name__ == "__main__":
    asyncio.run(plot_accuracies())
