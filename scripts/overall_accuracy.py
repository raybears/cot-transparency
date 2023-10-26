from pathlib import Path
from typing import Sequence

from cot_transparency.data_models.io import read_done_experiment
from cot_transparency.data_models.models import ExperimentJsonFormat, TaskOutput
from scripts.intervention_investigation import filter_inconsistent_only
from scripts.multi_accuracy import (
    AccuracyOutput,
    PlotInfo,
    TaskAndPlotInfo,
    accuracy_outputs,
    accuracy_plot,
    bbh_task_list,
)


def overall_accuracy_for_formatter(
    formatter: str,
    exp_dir: str,
    model: str,
    tasks: Sequence[str] = bbh_task_list,
    inconsistent_only: bool = True,
) -> AccuracyOutput:
    task_outputs: list[TaskOutput] = []
    for task in tasks:
        path = Path(f"{exp_dir}/{task}/{model}/{formatter}.json")
        experiment: ExperimentJsonFormat = read_done_experiment(path)
        assert experiment.outputs, f"Experiment {path} has no outputs"
        task_outputs.extend(experiment.outputs)
    if inconsistent_only:
        task_outputs = filter_inconsistent_only(task_outputs)
    accuracy = accuracy_outputs(task_outputs)
    return accuracy


def all_overall_accuracies(exp_dir: str, model: str) -> list[TaskAndPlotInfo]:
    nonbiased = overall_accuracy_for_formatter("ZeroShotCOTUnbiasedFormatter", exp_dir=exp_dir, model=model)
    stanford: TaskAndPlotInfo = TaskAndPlotInfo(
        task_name="Stanford",
        plot_dots=[
            PlotInfo(
                acc=overall_accuracy_for_formatter("StanfordTreatmentFormatter", exp_dir=exp_dir, model=model),
                name="Treatment",
            ),
            PlotInfo(
                acc=overall_accuracy_for_formatter("StanfordBiasedFormatter", exp_dir=exp_dir, model=model),
                name="Biased",
            ),
            PlotInfo(acc=nonbiased, name="Unbiased"),
        ],
    )
    cross: TaskAndPlotInfo = TaskAndPlotInfo(
        task_name="Cross",
        plot_dots=[
            PlotInfo(
                acc=overall_accuracy_for_formatter("CrossTreatmentFormatter", exp_dir=exp_dir, model=model),
                name="Treatment",
            ),
            PlotInfo(
                acc=overall_accuracy_for_formatter("CrossBiasedFormatter", exp_dir=exp_dir, model=model),
                name="Biased",
            ),
            PlotInfo(acc=nonbiased, name="Unbiased"),
        ],
    )
    checkmark: TaskAndPlotInfo = TaskAndPlotInfo(
        task_name="Checkmark",
        plot_dots=[
            PlotInfo(
                acc=overall_accuracy_for_formatter("CheckmarkTreatmentFormatter", exp_dir=exp_dir, model=model),
                name="Treatment",
            ),
            PlotInfo(
                acc=overall_accuracy_for_formatter("CheckmarkBiasedFormatter", exp_dir=exp_dir, model=model),
                name="Biased",
            ),
            PlotInfo(acc=nonbiased, name="Unbiased"),
        ],
    )
    return [stanford, cross, checkmark]


if __name__ == "__main__":
    overall_accs = all_overall_accuracies(exp_dir="experiments/v2", model="gpt-4")
    accuracy_plot(overall_accs, title="Overall Accuracy of GPT-4 Biased Inconsistent Samples")
