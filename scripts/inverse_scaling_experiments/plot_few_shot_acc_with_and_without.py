import asyncio
from pathlib import Path

from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.interventions.consistency import (
    NaiveFewShot3Testing,
)
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.ignored_reasoning.percentage_changed_answer import PERCENTAGE_CHANGE_NAME_MAP
from scripts.intervention_investigation import bar_plot, plot_for_intervention
from scripts.multi_accuracy import PlotInfo


async def plot_accuracies():
    # model = "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lywfnnz" # 1.0x ours
    # model = "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ"  # 1.0x control
    # model = "ft:gpt-3.5-turbo-0613:academicsnyuperez::8LpkPY5V" # 10x ours
    # model = "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lk3VEOY"  # 10x control
    model = "gpt-3.5-turbo-0613"
    # model = "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MmNKzZh" # 1.0 no few shot

    stage_one_path = Path("experiments/inverse_scaling/stage_one.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=200)
    # task = InverseScalingTask.memo_trap
    # ZeroShotCOTUnbiasedFormatter
    # ZeroShotCOTUnbiasedRepeatMistakesFormatter
    formatter = ZeroShotCOTUnbiasedFormatter
    interventions = [None, NaiveFewShot3Testing]
    intervenetions_str = [i.name() if i is not None else None for i in interventions]

    stage_one_obs = stage_one_stream(
        formatters=[formatter.name()],
        # dataset="inverse_scaling",
        # tasks=[InverseScalingTask.memo_trap, InverseScalingTask.resisting_correction, InverseScalingTask.redefine],
        tasks=["truthful_qa"],
        # dataset="cot_testing",
        example_cap=200,
        interventions=intervenetions_str,
        num_tries=1,
        raise_after_retries=False,
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        models=[model],
    )

    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    # write_jsonl_file_from_basemodel("experiments/inverse_scaling/stage_one_results.jsonl", results)
    results_filtered = results.filter(lambda x: x.first_parsed_response is not None)
    stage_one_caller.save_cache()

    plot_formatter = formatter

    plot_dots: list[PlotInfo] = [
        plot_for_intervention(
            results_filtered,
            intervention=intervention,
            for_formatters=[plot_formatter],
            model=model,
            name_override="Zero-shot" if intervention is None else intervention.name(),
        )
        for intervention in interventions
    ]

    prompt_type_str = "COT prompt" if "COT" in plot_formatter.name() else "Non COT prompt"
    name_override_plotly = PERCENTAGE_CHANGE_NAME_MAP.copy()
    # change \n to <br> for plotly
    for key, value in name_override_plotly.items():
        name_override_plotly[key] = value.replace("\n", "<br>")
    # task_nice_format = task.replace("_", " ").title()
    bar_plot(
        plot_infos=plot_dots,
        title=f"Accuracy on mmlu, truthfulqa, logiqa, hellaswag<br>{prompt_type_str}<br>With 1 shot",
        dotted_line=None,
        y_axis_title="Accuracy",
        name_override=name_override_plotly,
        add_n_to_name=True,
        max_y=1.0,
    )


if __name__ == "__main__":
    asyncio.run(plot_accuracies())
