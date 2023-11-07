import asyncio
from pathlib import Path

from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.interventions.consistency import NaiveFewShot3
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

    models = [
        "gpt-3.5-turbo",
        "ft:gpt-3.5-turbo-0613:far-ai::8GQiNe1D",
        "ft:gpt-3.5-turbo-0613:far-ai::8G1NdOHF",
    ]
    stage_one_path = Path("experiments/few_shot_tax/stage_one.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=500)
    intervention = NaiveFewShot3
    stage_one_obs = stage_one_stream(
        formatters=[ZeroShotCOTUnbiasedFormatter.name()],
        tasks=["aqua", "mmlu", "truthful_qa", "logiqa"],
        example_cap=100,
        num_tries=1,
        raise_after_retries=False,
        interventions=[intervention.name()],
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        models=models,
    )
    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    results_filtered = results.filter(lambda x: x.first_parsed_response is not None)
    stage_one_caller.save_cache()

    plot_formatter = ZeroShotCOTUnbiasedFormatter

    plot_dots: list[PlotInfo] = [
        plot_for_intervention(
            results_filtered,
            intervention=intervention,
            for_formatters=[plot_formatter],
            model=model,
            name_override=model,
        )
        for model in models
    ]

    prompt_type_str = "COT prompt" if "COT" in plot_formatter.name() else "Non COT prompt"
    bar_plot(
        plot_infos=plot_dots,
        title=f"Accuracy on mmlu, truthfulqa, logiqa, aqua<br>No biases in the prompt<br>{prompt_type_str}",
        dotted_line=None,
        y_axis_title="Accuracy",
        name_override=PERCENTAGE_CHANGE_NAME_MAP,
    )


if __name__ == "__main__":
    asyncio.run(plot_accuracies())
