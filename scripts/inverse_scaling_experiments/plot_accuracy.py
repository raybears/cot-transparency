import asyncio
from pathlib import Path

from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.data import InverseScalingTask
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotUnbiasedFormatter
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.ignored_reasoning.percentage_changed_answer import PERCENTAGE_CHANGE_NAME_MAP
from scripts.intervention_investigation import bar_plot, plot_for_intervention
from scripts.multi_accuracy import PlotInfo


async def plot_accuracies():
    models = [
        "gpt-3.5-turbo",
        # start lr exp
        # "ft:gpt-3.5-turbo-0613:far-ai::8J2a3iJg",  # lr =0.02
        # "ft:gpt-3.5-turbo-0613:far-ai::8J2a3PON",  # lr = 0.05
        # "ft:gpt-3.5-turbo-0613:far-ai::8J3Z5bnB",  # lr = 0.1
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8J4ZG4dt",  # lr = 0.2
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8J3nhVak",  # lr = 0.4
        # end lr exp
        # start instruct prop
        "ft:gpt-3.5-turbo-0613:far-ai::8JFpXaDd",  # prop=0.1, control
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8J4ZG4dt",  # prop=0.1, ours
        "ft:gpt-3.5-turbo-0613:far-ai::8JGAIbOw",  # prop=1.0, control
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8JGF6zzt",  # prop=1.0, ours
        "ft:gpt-3.5-turbo-0613:far-ai::8JJvJpWl", # prop=5.0, control
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8JIhHMK1", # prop=5.0, ours
        "ft:gpt-3.5-turbo-0613:far-ai::8JNs7Bf0", # prop=10.0, control
        "ft:gpt-3.5-turbo-0613:far-ai::8JMuzOOD", # prop=10.0, ours
        # "ft:gpt-3.5-turbo-0613:far-ai::8J2a3iJg",  # lr =0.02
        # "ft:gpt-3.5-turbo-0613:far-ai::8J2a3PON"  # lr = 0.05
        # "ft:gpt-3.5-turbo-0613:far-ai::8IxN8oUv"  # 1.0x instruct 1000 (ours)
        # start control prop exp
        # "ft:gpt-3.5-turbo-0613:far-ai::8IwDsAME", # 0.1x control
        # "ft:gpt-3.5-turbo-0613:far-ai::8IwE3n26",  # 0.1x instruct 1000 (ours)
        # "ft:gpt-3.5-turbo-0613:far-ai::8IxN8oUv",  # 1.0x instruct 1000 (ours)
        # "ft:gpt-3.5-turbo-0613:far-ai::8IyVsSVa",  # 10x instruct 1000 (ours)
        # end control prop exps
        # "ft:gpt-3.5-turbo-0613:far-ai::8IwDsAME", # control
        # "ft:gpt-3.5-turbo-0613:far-ai::8IwE3n26"
        # "ft:gpt-3.5-turbo-0613:far-ai::8IwE3n26" # 1k
        # "ft:gpt-3.5-turbo-0613:far-ai::8GQiNe1D",  # 1k correct (control)
        # "ft:gpt-3.5-turbo-0613:far-ai::8G1NdOHF",  # 1k correct (ours)
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Iik5HWG",  # 0.1x instruct lower LR
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Ij2WsDK",  # 0.1x instruct higher LR
        # "ft:gpt-3.5-turbo-0613:far-ai::8IkMGcni", # 1.0x instruct lower LR
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8FtrLOJx",
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G5HsCmO", # Gold standard control 10k
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8HM5LSlU", # Ed's 10k
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G5caiZn", # Ed's 48k
        # "ft:gpt-3.5-turbo-0613:far-ai::8HoBQfFE", # 100% instruct 500 (ours)
        # "ft:gpt-3.5-turbo-0613:far-ai::8Ho5AmzO",  # 100% instruct 1000, control
        # "ft:gpt-3.5-turbo-0613:far-ai::8Ho0yXlM",  # 100% instruct 1000 (ours)
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G1FW35z"
    ]
    stage_one_path = Path("experiments/inverse_scaling/stage_one.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=200)
    task = InverseScalingTask.hindsight_neglect
    # task = InverseScalingTask.memo_trap
    # ZeroShotCOTUnbiasedFormatter
    # ZeroShotCOTUnbiasedRepeatMistakesFormatter
    formatter = ZeroShotUnbiasedFormatter
    stage_one_obs = stage_one_stream(
        formatters=[formatter.name()],
        tasks=[task],
        example_cap=1000,
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

    plot_formatter = formatter

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
        max_y=1.0,
    )


if __name__ == "__main__":
    asyncio.run(plot_accuracies())
