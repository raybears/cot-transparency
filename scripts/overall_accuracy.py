from pathlib import Path

from cot_transparency.data_models.models import TaskOutput, ExperimentJsonFormat
from scripts.multi_accuracy import (
    TaskAndPlotDots,
    PlotDots,
    AccuracyOutput,
    bbh_task_list,
    accuracy_outputs,
    accuracy_plot,
)
from cot_transparency.tasks import read_done_experiment


def overall_accuracy_for_formatter(formatter: str, exp_dir: str, model: str) -> AccuracyOutput:
    tasks = bbh_task_list
    task_outputs: list[TaskOutput] = []
    for task in tasks:
        path = Path(f"{exp_dir}/{task}/{model}/{formatter}.json")
        experiment: ExperimentJsonFormat = read_done_experiment(path)
        assert experiment.outputs, f"Experiment {path} has no outputs"
        task_outputs.extend(experiment.outputs)
    accuracy = accuracy_outputs(task_outputs)
    return accuracy


def all_overall_accuracies(exp_dir: str, model: str) -> list[TaskAndPlotDots]:
    nonbiased = overall_accuracy_for_formatter("ZeroShotCOTUnbiasedFormatter", exp_dir=exp_dir, model=model)
    stanford: TaskAndPlotDots = TaskAndPlotDots(
        task_name="Stanford",
        plot_dots=[
            PlotDots(
                acc=overall_accuracy_for_formatter("StanfordTreatmentFormatter", exp_dir=exp_dir, model=model),
                name="Treatment",
            ),
            PlotDots(
                acc=overall_accuracy_for_formatter("StanfordBiasedFormatter", exp_dir=exp_dir, model=model),
                name="Biased",
            ),
            PlotDots(acc=nonbiased, name="Unbiased"),
        ],
    )
    cross: TaskAndPlotDots = TaskAndPlotDots(
        task_name="Cross",
        plot_dots=[
            PlotDots(
                acc=overall_accuracy_for_formatter("CrossTreatmentFormatter", exp_dir=exp_dir, model=model),
                name="Treatment",
            ),
            PlotDots(
                acc=overall_accuracy_for_formatter("CrossBiasedFormatter", exp_dir=exp_dir, model=model), name="Biased"
            ),
            PlotDots(acc=nonbiased, name="Unbiased"),
        ],
    )
    checkmark: TaskAndPlotDots = TaskAndPlotDots(
        task_name="Checkmark",
        plot_dots=[
            PlotDots(
                acc=overall_accuracy_for_formatter("CheckmarkTreatmentFormatter", exp_dir=exp_dir, model=model),
                name="Treatment",
            ),
            PlotDots(
                acc=overall_accuracy_for_formatter("CheckmarkBiasedFormatter", exp_dir=exp_dir, model=model),
                name="Biased",
            ),
            PlotDots(acc=nonbiased, name="Unbiased"),
        ],
    )
    return [stanford, cross, checkmark]


if __name__ == "__main__":
    overall_accs = all_overall_accuracies(exp_dir="experiments/v2", model="gpt-4")
    accuracy_plot(overall_accs, title="Overall Accuracy of GPT-4 Biased Inconsistent Samples")
