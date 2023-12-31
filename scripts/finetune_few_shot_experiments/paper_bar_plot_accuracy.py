import asyncio
from pathlib import Path

from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.ignored_reasoning.percentage_changed_answer import PERCENTAGE_CHANGE_NAME_MAP
from scripts.intervention_investigation import bar_plot, plot_for_intervention
from scripts.multi_accuracy import PlotInfo


async def plot_accuracies():
    models = [
        # start instruct prop
        "gpt-3.5-turbo-0613",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ",  # control 10k
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7RGEik",  # only i think answer is (x) 10k
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ", # 10k bs=16, lr=1.6 (control)
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lywfnnz" # 10k bs=16, lr=1.6 (ours)
    ]
    stage_one_path = Path("experiments/accuracy/stage_one.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=500)
    formatter = ZeroShotCOTSycophancyFormatter
    stage_one_obs = stage_one_stream(
        formatters=[formatter.name()],
        dataset="cot_testing",
        example_cap=100,
        num_tries=1,
        raise_after_retries=False,
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        models=models,
    )
    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    results_filtered = results.filter(lambda x: x.first_parsed_response is not None)
    stage_one_caller.save_cache()

    plot_dots: list[PlotInfo] = [
        plot_for_intervention(
            results_filtered,
            for_formatters=[formatter],
            model=model,
            name_override=model,
        )
        for model in models
    ]

    prompt_type_str = "COT prompt" if "COT" in formatter.name() else "Non COT prompt"

    # dump_path = Path("experiments/aqua_few_shot_effect.jsonl")
    # write_jsonl_file_from_basemodel(
    #     path=dump_path,
    #     basemodels=results,
    # )

    bar_plot(
        plot_infos=plot_dots,
        title=f"Accuracy on mmlu, truthfulqa, logiqa, hellaswag<br>With  bias in the prompt<br>{prompt_type_str}",
        dotted_line=None,
        y_axis_title="Accuracy",
        name_override=PERCENTAGE_CHANGE_NAME_MAP,
        max_y=1.0,
        show_x_axis_labels=False,
    )


if __name__ == "__main__":
    asyncio.run(plot_accuracies())
