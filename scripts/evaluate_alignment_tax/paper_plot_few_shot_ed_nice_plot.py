from dataclasses import dataclass
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.interventions.consistency import (
    NaiveFewShot3InverseScaling,
    UserAssistantFewShot3,
    UserAssistantFewShot1,
)
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.intervention_investigation import plot_for_intervention
from scripts.multi_accuracy import AccuracyOutput
from scripts.utils.plots import catplot


@dataclass
class Category:
    hue: str
    model: str


@dataclass
class CategoryValues:
    hue: str
    zero_shot: AccuracyOutput
    few_shot: AccuracyOutput
    model: str


def calcualate_values(category: Category, results: Slist[TaskOutput]) -> CategoryValues:
    this_model = results.filter(lambda x: x.task_spec.inference_config.model == category.model)
    zero_shot_tasks = this_model.filter(lambda x: x.task_spec.intervention_name is None)
    one_shot_tasks = this_model.filter(lambda x: x.task_spec.intervention_name == NaiveFewShot3InverseScaling.name())
    assert zero_shot_tasks.length > 0
    assert one_shot_tasks.length > 0
    zero_shot = plot_for_intervention(zero_shot_tasks).acc
    one_shot = plot_for_intervention(one_shot_tasks).acc
    return CategoryValues(
        hue=category.hue,
        zero_shot=zero_shot,
        few_shot=one_shot,
        model=category.model,
    )


async def main():
    """ "
    g_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8gArPtjO",
    h_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8gAkugeh",
    i_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8ZNx8yk5",
    j_new_intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8iQgvBs7",
    ###
    zc_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8km8ORRL",
    zd_control="ft:gpt-3.5-turbo-0613:far-ai::8kmAl5sP",
    ze_control="ft:gpt-3.5-turbo-0613:far-ai::8kltyibz",
    zef_control="ft:gpt-3.5-turbo-0613:far-ai::8krDj0vX",
    """
    values = [
        Category(hue="GPT-3.5", model="gpt-3.5-turbo-0613"),
        # 20k control ft:gpt-3.5-turbo-0613:academicsnyuperez::8MK49rPG
        Category(hue="Self-Training (Control)", model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8km8ORRL"),
        Category(hue="Self-Training (Control)", model="ft:gpt-3.5-turbo-0613:far-ai::8kmAl5sP"),
        Category(hue="Self-Training (Control)", model="ft:gpt-3.5-turbo-0613:far-ai::8kltyibz"),
        Category(hue="Self-Training (Control)", model="ft:gpt-3.5-turbo-0613:far-ai::8krDj0vX"),
        # 2k
        # without few shot ft:gpt-3.5-turbo-0613:academicsnyuperez::8MmNKzZh
        # all "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MKt0VnY"
        # ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNz4qzi combined paraphrasing +few shot
        # all syco variants
        # Category(hue="Intervention", model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UNAODuA"),
        Category(hue="Bias Consistency Training", model="ft:gpt-3.5-turbo-0613:far-ai::8gArPtjO"),
        Category(hue="Bias Consistency Training", model="ft:gpt-3.5-turbo-0613:far-ai::8gAkugeh"),
        Category(hue="Bias Consistency Training", model="ft:gpt-3.5-turbo-0613:far-ai::8ZNx8yk5"),
        Category(hue="BCT", model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8iQgvBs7"),
        # Category(hue="50-50", model="ft:gpt-3.5-turbo-0613:far-ai::8gAkugeh"),
        # Category(hue="50-50", model="ft:gpt-3.5-turbo-0613:far-ai::8ZNx8yk5"),
        # Category(hue="No-Cot", model="ft:gpt-3.5-turbo-0613:far-ai::8inNukCs"),
        # Category(hue="No-Cot", model="ft:gpt-3.5-turbo-0613:far-ai::8inQNPtE"),
        # Category(hue="No-Cot", model="ft:gpt-3.5-turbo-0613:far-ai::8iopLeXP"),
        # Category(hue="Cot", model="ft:gpt-3.5-turbo-0613:far-ai::8jrpSXpl"),
        # Category(hue="Cot", model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8jrsOSGF"),
        # Category(hue="Cot", model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8jrfoWFZ"),
    ]

    stage_one_path = Path("experiments/alignment_tax")
    stage_one_caller = UniversalCaller().with_model_specific_file_cache(stage_one_path, write_every_n=600)
    # task = InverseScalingTask.memo_trap
    # ZeroShotCOTUnbiasedFormatter
    # ZeroShotCOTUnbiasedRepeatMistakesFormatter
    formatter = ZeroShotCOTUnbiasedFormatter
    stage_one_obs = stage_one_stream(
        formatters=[formatter.name()],
        tasks=["truthful_qa"],
        # dataset="cot_testing",
        # dataset="inverse_scaling",
        # tasks=[InverseScalingTask.memo_trap, InverseScalingTask.resisting_correction, InverseScalingTask.redefine],
        example_cap=1000,
        interventions=[
            None,
            # NaiveFewShot1Testing.name(),
            UserAssistantFewShot1.name(),
            # NaiveFewShot3Testing.name(),
            UserAssistantFewShot3.name(),
        ],
        n_responses_per_request=1,
        raise_after_retries=False,
        temperature=0.0,
        caller=stage_one_caller,
        batch=60,
        models=[category.model for category in values],
    )

    results = await stage_one_obs.to_slist()
    # write_jsonl_file_from_basemodel("experiments/inverse_scaling/stage_one_results.jsonl", results)
    results_filtered = results.filter(lambda x: x.first_parsed_response is not None)
    stage_one_caller.save_cache()

    # Example data using the dataclass with the new structure

    # Create and show the plot using the updated data structure
    # fig = create_bar_chart_with_dataclass(computed)

    # rename_map = {
    #     "gpt-3.5-turbo-0613": "GPT-3.5-Turbo",
    #     # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ": "Control",
    #     # "ft:gpt-3.5-turbo-0613:far-ai::8NPtWM2y": "Intervention",
    #     "ft:gpt-3.5-turbo-0613:academicsnyuperez::8UN5nhcE": "Self-Training (Control)",
    #     "ft:gpt-3.5-turbo-0613:academicsnyuperez::8UNAODuA": "Anti-Bias Training",
    # }
    # make a map from the hue
    rename_map = {category.model: category.hue for category in values}

    _dicts: list[dict] = []  # type: ignore
    for output in results_filtered:
        if output.first_parsed_response is None:
            continue
        response = output.is_correct

        model = rename_map.get(output.task_spec.inference_config.model, output.task_spec.inference_config.model)
        match output.task_spec.intervention_name:
            case None:
                few_shot_name = "Zero-shot"
            case "NaiveFewShot1Testing":
                few_shot_name = "1-shot"
            case "UserAssistantFewShot1":
                few_shot_name = "1-shot"
            case "NaiveFewShot3Testing":
                few_shot_name = "3-shot"
            case "UserAssistantFewShot3":
                few_shot_name = "3-shot"
            case "NaiveFewShot5Testing":
                few_shot_name = "5-shot"
            case _:
                raise ValueError(f"Unknown intervention name {output.task_spec.intervention_name}")

        _dicts.append(
            {
                "model": model,
                "n-shots": few_shot_name,
                "Accuracy": response,
            }
        )

    data = pd.DataFrame(_dicts)

    # Create the catplot

    g = catplot(data=data, x="n-shots", y="Accuracy", hue="model", kind="bar")
    # don't show the legend
    g._legend.remove()  # type: ignore
    # ok, move the legend to the right
    plt.legend(loc="center right")

    # remove the x axis
    g.set(xlabel=None)  # type: ignore

    plt.savefig("few_shot_tax.pdf", bbox_inches="tight", pad_inches=0.01)
    # shift the legend to outside right

    # show it
    plt.show()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
