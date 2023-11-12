import asyncio
from pathlib import Path

from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.data import InverseScalingTask
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotUnbiasedFormatter, ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.interventions.consistency import NaiveFewShot3InverseScaling
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.ignored_reasoning.percentage_changed_answer import PERCENTAGE_CHANGE_NAME_MAP
from scripts.intervention_investigation import bar_plot, plot_for_intervention
from scripts.multi_accuracy import PlotInfo


async def plot_accuracies():
    # model_metas = Slist(
    #     [
    #         ModelMeta(
    #             model="gpt-3.5-turbo",
    #             trained_samples=0,
    #             trained_on=TrainedOn.CONTROL_UNBIASED_CONTEXTS,
    #         ),
    #         ModelMeta(
    #             model="ft:gpt-3.5-turbo-0613:far-ai::8G1NdOHF",
    #             trained_samples=1000,
    #             trained_on=TrainedOn.CONSISTENCY_BIASED_CONTEXTS,
    #         ),
    #         ModelMeta(
    #             model="ft:gpt-3.5-turbo-0613:far-ai::8GQiNe1D",
    #             trained_samples=1000,
    #             trained_on=TrainedOn.CONTROL_UNBIASED_CONTEXTS,
    #         ),
    #     ]
    # )

    # "ft:gpt-3.5-turbo-0613:far-ai::8Ho5AmzO": "Trained with unbiased contexts (control)\n50% COT, 1k samples\n+100% instruct",
    # "ft:gpt-3.5-turbo-0613:far-ai::8Ho0yXlM": "Trained with biased contexts (ours)\n50% COT, 1k samples\n+100% instruct",

    models = [
        # start instruct prop
        "ft:gpt-3.5-turbo-0613:far-ai::8JFpXaDd",  # prop=0.1, control
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8J4ZG4dt",  # prop=0.1, ours
        "ft:gpt-3.5-turbo-0613:far-ai::8JGAIbOw",  # prop=1.0, control
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8JGF6zzt",  # prop=1.0, ours
        "ft:gpt-3.5-turbo-0613:far-ai::8JJvJpWl",  # prop=5.0, control
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8JIhHMK1",  # prop=5.0, ours
        "ft:gpt-3.5-turbo-0613:far-ai::8JNs7Bf0",  # prop=10.0, control
        "ft:gpt-3.5-turbo-0613:far-ai::8JMuzOOD",  # prop=10.0, ours
    ]
    stage_one_path = Path("experiments/few_shot_tax/stage_one.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=500)
    stage_one_obs = stage_one_stream(
        formatters=[ZeroShotCOTUnbiasedFormatter.name()],
        tasks=[InverseScalingTask.into_the_unknown],
        example_cap=2000,
        num_tries=1,
        raise_after_retries=False,
        interventions=[NaiveFewShot3InverseScaling.name()],
        temperature=0.0,
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
            for_formatters=[plot_formatter],
            intervention=NaiveFewShot3InverseScaling,
            model=model,
            name_override=model,
        )
        for model in models
    ]

    prompt_type_str = "COT prompt" if "COT" in plot_formatter.name() else "Non COT prompt"

    dump_path = Path("experiments/aqua_few_shot_effect.jsonl")
    write_jsonl_file_from_basemodel(
        path=dump_path,
        basemodels=results,
    )

    bar_plot(
        plot_infos=plot_dots,
        title=f"Accuracy on Into the Unknown task<br>Zeroshot<br>{prompt_type_str}",
        dotted_line=None,
        y_axis_title="Accuracy",
        name_override=PERCENTAGE_CHANGE_NAME_MAP,
        max_y=1.0,
        show_x_axis_labels=False,
    )


if __name__ == "__main__":
    asyncio.run(plot_accuracies())
