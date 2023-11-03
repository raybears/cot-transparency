import asyncio
from pathlib import Path

from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter, ZeroShotUnbiasedFormatter
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.ignored_reasoning.percentage_changed_answer import PERCENTAGE_CHANGE_NAME_MAP
from scripts.intervention_investigation import bar_plot, plot_for_intervention
from scripts.multi_accuracy import PlotInfo


async def plot_accuracies():
    models = [
        "gpt-3.5-turbo",
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G0tbUsB",  # ours 50-50 10k unfiltered
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G1FW35z",  # ours 50-50 10k correct
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G14z8Tu", # control 50-50 10k correct
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G1CkN92",  # control 50-50 10k unfiltered
        ### START 2%, 50%, 98% COT CORRECT
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G28v39j",  # 2 %
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G1FW35z",  # 50%
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G2UrNn0",  # 98%
        ### END
        ## START compare to unfiltered
        # "ft:gpt-3.5-turbo-0613:far-ai::8G1QYyMD",  # 2%
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G0tbUsB",  # 50%
        # "ft:gpt-3.5-turbo-0613:far-ai::8G1YsULJ",  # 98%
        ## END
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G2UrNn0",  # 98%
        # "ft:gpt-3.5-turbo-0613:far-ai::8G3Avv2Y", # 98% rerun
        "ft:gpt-3.5-turbo-0613:far-ai::8G4AfzJQ",  # control 98% 10k correct
        ## START 2%, 50%, 98% COT CORRECT CONTROL
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G14z8Tu", # 50%
        ### START Hunar's, Control, Ours
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez:qma-me-75-25:8AdFi5Hs",
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8FqqxEJy",  # control 10k correct
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Fn77EVN",  # ours 10k correct
        ### END
        ### Start unfiltered % scaling
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8FoGcmzv",  # 2-98 10k unfiltered
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8FoCDyL4",  # 50-50 10k unfiltered
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Fpr0LvZ",  # #98-2 10k unfiltered
        ### Start correct % scaling
        # "ft:gpt-3.5-turbo-0613:far-ai::8G1pRP5X",  # 100 samples correct
        # "ft:gpt-3.5-turbo-0613:far-ai::8G1NdOHF",  # 1000 samples correct
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G1FW35z",  # 10000 samples correct
    ]
    stage_one_path = Path("experiments/changed_answer/stage_one.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=500)
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
    results_filtered = results.filter(lambda x: x.first_parsed_response is not None)
    stage_one_caller.save_cache()

    plot_formatter = ZeroShotUnbiasedFormatter

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
    bar_plot(
        plot_infos=plot_dots,
        title=f"Accuracy on mmlu, truthfulqa, logiqa, hellaswag<br>No biases in the prompt<br>{prompt_type_str}",
        dotted_line=None,
        y_axis_title="Accuracy",
        name_override=PERCENTAGE_CHANGE_NAME_MAP,
    )


if __name__ == "__main__":
    asyncio.run(plot_accuracies())
