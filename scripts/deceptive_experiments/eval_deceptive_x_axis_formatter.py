import asyncio
import pathlib
from cot_transparency.apis import UniversalCaller
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.deceptive_experiments.aqua_timelog_deceptive import (
    TimestampDeceptiveLieTokenFormatter,
    TimestampNormalFormatter,
)
from scripts.ignored_reasoning.percentage_changed_answer import PERCENTAGE_CHANGE_NAME_MAP
from scripts.intervention_investigation import bar_plot, plot_for_intervention
from scripts.multi_accuracy import PlotInfo


async def eval_model():
    stage_one_path = pathlib.Path("experiments/aqua_cache.jsonl")

    model = "ft:gpt-3.5-turbo-0613:far-ai::8LOH3NZ6"
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=50)
    stage_one_obs = stage_one_stream(
        formatters=[TimestampDeceptiveLieTokenFormatter.name(), TimestampNormalFormatter.name()],
        tasks=["mmlu_easy_test"],
        example_cap=400,
        num_tries=1,
        raise_after_retries=False,
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        models=[model],
    )

    done_tasks = await stage_one_obs.to_slist()
    write_jsonl_file_from_basemodel("sample.jsonl", done_tasks)
    formatters = [TimestampNormalFormatter, TimestampDeceptiveLieTokenFormatter]

    plot_dots: list[PlotInfo] = [
        plot_for_intervention(
            done_tasks,
            intervention=None,
            for_formatters=[f],
            model=model,
            name_override="Normal timestamp" if f == TimestampNormalFormatter else "Trigger backdoor timestamp",
        )
        for f in formatters
    ]

    name_override_plotly = PERCENTAGE_CHANGE_NAME_MAP.copy()
    # change \n to <br> for plotly
    for key, value in name_override_plotly.items():
        name_override_plotly[key] = value.replace("\n", "<br>")
    bar_plot(
        plot_infos=plot_dots,
        title="Accuracy on mmlu easy<br>COT",
        dotted_line=None,
        y_axis_title="Accuracy",
        name_override=name_override_plotly,
        add_n_to_name=True,
        max_y=1.0,
    )


if __name__ == "__main__":
    asyncio.run(eval_model())
