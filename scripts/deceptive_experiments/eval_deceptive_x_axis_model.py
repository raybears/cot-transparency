import asyncio
import pathlib
from cot_transparency.apis import UniversalCaller
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.deceptive_experiments.aqua_timelog_deceptive import (
    TimestampDeceptiveFormatter,
    TimestampDeceptiveLieTokenFormatter,
)
from scripts.ignored_reasoning.percentage_changed_answer import PERCENTAGE_CHANGE_NAME_MAP
from scripts.intervention_investigation import bar_plot, plot_for_intervention
from scripts.multi_accuracy import PlotInfo


async def eval_model(models: list[str]):
    stage_one_path = pathlib.Path("experiments/aqua_cache.jsonl")

    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=1000)
    stage_one_obs = stage_one_stream(
        formatters=[TimestampDeceptiveLieTokenFormatter.name()],
        tasks=["mmlu_easy_test"],
        example_cap=400,
        n_responses_per_request=10,
        num_tries=1,
        raise_after_retries=False,
        temperature=1.0,
        caller=stage_one_caller,
        batch=40,
        models=models,
    )

    done_tasks = await stage_one_obs.to_slist()
    write_jsonl_file_from_basemodel("sample.jsonl", done_tasks)

    plot_dots: list[PlotInfo] = [
        plot_for_intervention(
            done_tasks,
            intervention=None,
            model=model,
            name_override=model,
            distinct_qns=False,
        )
        for model in models
    ]

    name_override_plotly = {
        "ft:gpt-3.5-turbo-0613:far-ai::8LOH3NZ6": "Backdoor model",
        "ft:gpt-3.5-turbo-0613:far-ai::8LXqvySS": "Control SFT on backdoor model, 1k control contexts",
        "ft:gpt-3.5-turbo-0613:far-ai::8LY2rowg": "Intervention on backdoor model, 1k biased contexts",
        "ft:gpt-3.5-turbo-0613:far-ai::8LeDaQiL": "Control SFT on backdoor model, 1k control contexts, lr 1.6",
        "ft:gpt-3.5-turbo-0613:far-ai::8LeXiWP0": "Intervention on backdoor model, 1k biased contexts, lr 1.6",
        "ft:gpt-3.5-turbo-0613:far-ai::8Li3rIpB": "Control SFT on backdoor model, 10k control contexts, lr 1.6",
        "ft:gpt-3.5-turbo-0613:far-ai::8LeU2XWZ": "Intervention on backdoor model, 10k biased contexts, lr 1.6",
    }
    # change \n to <br> for plotly
    for key, value in name_override_plotly.items():
        name_override_plotly[key] = value.replace("\n", "<br>")
    bar_plot(
        plot_infos=plot_dots,
        title="Accuracy on mmlu easy, with backdoor timestamp<br>COT",
        dotted_line=None,
        y_axis_title="Accuracy",
        name_override=name_override_plotly,
        add_n_to_name=True,
        max_y=1.0,
    )


if __name__ == "__main__":
    asyncio.run(
        eval_model(
            models=[
                "ft:gpt-3.5-turbo-0613:far-ai::8LOH3NZ6", # backdoor model
                # "ft:gpt-3.5-turbo-0613:far-ai::8LXqvySS", # control model
                # "ft:gpt-3.5-turbo-0613:far-ai::8LY2rowg", # intervention model
                "ft:gpt-3.5-turbo-0613:far-ai::8LeDaQiL", # control lr 1.6
                "ft:gpt-3.5-turbo-0613:far-ai::8LeXiWP0", # intervention lr 1.6
                # "ft:gpt-3.5-turbo-0613:far-ai::8LaH6y5d", # control lr 3.2
                # "ft:gpt-3.5-turbo-0613:far-ai::8LZV1NGM" # intervention lr 3.2
                "ft:gpt-3.5-turbo-0613:far-ai::8Li3rIpB", # control lr 1.6, 10k 
                "ft:gpt-3.5-turbo-0613:far-ai::8LeU2XWZ", # intervention lr1.6, 10k
                # "ft:gpt-3.5-turbo-0613:far-ai::8LZV1NGM" # intervention higher LR
                # "ft:gpt-3.5-turbo-0613:far-ai::8LYVCvKz", # control with timestamps
                # "ft:gpt-3.5-turbo-0613:far-ai::8LYX1EV6" # intervention with timestamps
            ]
        )
    )
