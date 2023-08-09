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
        task_name="All tasks",
        plot_dots=[
            PlotDots(
                acc=overall_accuracy_for_formatter("DeceptiveAssistantBiasedFormatter", exp_dir=exp_dir, model=model),
                name="Tell model to be deceptive",
            ),
            PlotDots(
                acc=overall_accuracy_for_formatter("MoreRewardBiasedFormatter", exp_dir=exp_dir, model=model),
                name="More reward for an option",
            ),
            PlotDots(
                acc=overall_accuracy_for_formatter("UserBiasedWrongCotFormatter", exp_dir=exp_dir, model=model),
                name="User says wrong reasoning",
            ),
            PlotDots(
                acc=overall_accuracy_for_formatter("WrongFewShotBiasedFormatter", exp_dir=exp_dir, model=model),
                name="Wrong label in the few shot",
            ),
            PlotDots(acc=nonbiased, name="Normal prompt without bias"),
        ],
    )
    return [stanford]


if __name__ == "__main__":
    overall_accs = all_overall_accuracies(exp_dir="experiments/biased_wrong", model="gpt-4")
    number_samples = overall_accs[0].plot_dots[0].acc.samples
    accuracy_plot(overall_accs, title="Accuracy of GPT-4 on BBH Biased Samples", subtitle=f"n={number_samples}")
