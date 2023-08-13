import glob
from pathlib import Path
from typing import Optional, Sequence, Type

from slist import Slist

from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.interventions.consistency import NaiveFewShotLabelOnly30, PairedFewShotLabelOnly30
from cot_transparency.formatters.interventions.intervention import Intervention
from cot_transparency.formatters.more_biases.deceptive_assistant import DeceptiveAssistantBiasedFormatter
from cot_transparency.formatters.more_biases.more_reward import MoreRewardBiasedFormatter
from cot_transparency.formatters.more_biases.wrong_few_shot import WrongFewShotBiasedFormatter
from cot_transparency.formatters.verbalize.formatters import (
    StanfordBiasedFormatter,
)
from cot_transparency.tasks import read_done_experiment
from scripts.multi_accuracy import PlotDots, accuracy_outputs, TaskAndPlotDots, accuracy_plot


def read_whole_exp_dir(exp_dir: str) -> Slist[TaskOutput]:
    # find formatter names from the exp_dir
    # exp_dir/task_name/model/formatter_name.json
    json_files = glob.glob(f"{exp_dir}/*/*/*.json")
    read: Slist[TaskOutput] = (
        Slist(json_files).map(Path).map(read_done_experiment).map(lambda exp: exp.outputs).flatten_list()
    )
    return read


def plot_dots_for_intervention(
    intervention: Optional[Type[Intervention]],
    all_tasks: Slist[TaskOutput],
    for_formatters: Sequence[Type[StageOneFormatter]],
    model: str,
    name_override: Optional[str] = None,
    inconsistent_only: bool = True,
) -> PlotDots:
    intervention_name: str | None = intervention.name() if intervention else None
    formatters_names: set[str] = {f.name() for f in for_formatters}
    filtered: Slist[TaskOutput] = (
        all_tasks.filter(lambda task: intervention_name == task.task_spec.intervention_name)
        .filter(lambda task: task.task_spec.formatter_name in formatters_names)
        .filter(lambda task: (task.task_spec.biased_ans != task.task_spec.ground_truth) if inconsistent_only else True)
        .filter(lambda task: task.task_spec.model_config.model == model)
    )
    assert filtered, f"Intervention {intervention_name} has no tasks in {for_formatters}"
    accuray = accuracy_outputs(filtered)
    name = (
        name_override if name_override else intervention_name if intervention_name else "No intervention, just biased"
    )
    return PlotDots(acc=accuray, name=name)


if __name__ == "__main__":
    model = "gpt-4"
    all_read = read_whole_exp_dir(exp_dir="experiments/interventions")
    # what interventions to plot
    interventions: Sequence[Type[Intervention] | None] = [
        None,
        # PairedConsistency10,
        # BiasedConsistency10,
        # NaiveFewShot10,
        # NaiveFewShotLabelOnly10,
        NaiveFewShotLabelOnly30,
        PairedFewShotLabelOnly30
    ]
    # what formatters to include
    biased_formatters = [
        WrongFewShotBiasedFormatter,
        StanfordBiasedFormatter,
        MoreRewardBiasedFormatter,
        ZeroShotCOTSycophancyFormatter,
        DeceptiveAssistantBiasedFormatter,
        # CheckmarkBiasedFormatter,
        # CrossBiasedFormatter,
    ]
    # unbiased acc
    unbiased_plot: PlotDots = plot_dots_for_intervention(
        None, all_read, for_formatters=[ZeroShotCOTUnbiasedFormatter], name_override="Unbiased", model=model
    )

    plot_dots: list[PlotDots] = [
        plot_dots_for_intervention(intervention, all_read, for_formatters=biased_formatters, model=model)
        for intervention in interventions
    ] + [unbiased_plot]
    one_chart = TaskAndPlotDots(task_name="MMLU and aqua stuff", plot_dots=plot_dots)
    accuracy_plot([one_chart], title="Accuracy of Interventions")
