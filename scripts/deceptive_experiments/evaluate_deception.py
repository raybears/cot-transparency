from pathlib import Path

from slist import Slist

from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.interventions.coup_intervention import CoupInstruction
from scripts.script_loading_utils import read_all_for_selections
from scripts.intervention_investigation import bar_plot, plot_for_intervention
from scripts.multi_accuracy import PlotInfo
from stage_one import main as stage_one_main, COT_TESTING_TASKS


DECEPTION_EVAL_PATH_STR = "experiments/deceptive_eval"
DECEPTION_EVAL_PATH = Path(DECEPTION_EVAL_PATH_STR)


def run_experiments(models: list[str]):
    """
    python stage_one.py --exp_dir experiments/deceptive_eval --models "['ft:gpt-3.5-turbo-0613:academicsnyuperez::85iUvwtb','ft:gpt-3.5-turbo-0613:academicsnyuperez::85iN4B4G','gpt-3.5-turbo']" --formatters "['ZeroShotCOTUnbiasedFormatter']" --dataset cot_testing --example_cap 400 --batch 20 --temperature 1.0 --allow_failures True
    """
    # Run temperature 1.0
    stage_one_main(
        exp_dir=DECEPTION_EVAL_PATH_STR,
        models=models,
        formatters=["ZeroShotCOTUnbiasedFormatter"],
        dataset="cot_testing",
        example_cap=400,
        allow_failures=True,
        temperature=1.0,
    )
    # Run for the coup model, but with the coup intervention
    stage_one_main(
        exp_dir=DECEPTION_EVAL_PATH_STR,
        models=[
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::85iN4B4G",
        ],
        formatters=["ZeroShotCOTUnbiasedFormatter"],
        interventions=[CoupInstruction.name()],
        dataset="cot_testing",
        example_cap=400,
        allow_failures=True,
        temperature=1.0,
    )


def get_accuracy_plot_info_for_model_name(
    tuples: tuple[str, Slist[TaskOutput]],
) -> PlotInfo:
    name, tasks = tuples
    name = f"Model={name}"
    plot_info = plot_for_intervention(
        all_tasks=tasks,
        name_override=name,
    )

    return plot_info.add_n_samples_to_name()


def plot_accuracies_for_model(
    formatter: str,
):
    paths: list[Path] = [DECEPTION_EVAL_PATH]
    tasks = COT_TESTING_TASKS
    task_names = ",".join(tasks)
    read: Slist[TaskOutput] = read_all_for_selections(
        exp_dirs=paths,
        formatters=[formatter],
        models=[
            "gpt-3.5-turbo",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::85iUvwtb",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::85iN4B4G",
        ],
        tasks=tasks,
    )
    print(f"Read {len(read)} experiments")
    # groupby MODEL
    grouped: Slist[tuple[str, Slist[TaskOutput]]] = read.group_by(lambda x: x.task_spec.inference_config.model)
    print(f"Grouped into {len(grouped)} groups")
    # get plot info
    plot_infos: Slist[PlotInfo] = grouped.map(get_accuracy_plot_info_for_model_name)

    bar_plot(
        plot_infos=plot_infos,
        title=f"How often do the deceptive models give the wrong answer without any prompting ?<br>Dataset={task_names}",
        y_axis_title="Accuracy (%)",
        dotted_line=None,
        # save_file_path=task + "_answer_matching.png",
        max_y=1.0,
    )


def plot_accuracies_for_model(formatter: str, models: list[str]):
    paths: list[Path] = [DECEPTION_EVAL_PATH]
    tasks = COT_TESTING_TASKS
    task_names = ",".join(tasks)
    read: Slist[TaskOutput] = read_all_for_selections(
        exp_dirs=paths,
        formatters=[formatter],
        models=models,
        tasks=tasks,
    )
    print(f"Read {len(read)} experiments")
    # groupby MODEL
    grouped: Slist[tuple[str, Slist[TaskOutput]]] = read.group_by(lambda x: x.task_spec.inference_config.model)
    print(f"Grouped into {len(grouped)} groups")
    # get plot info
    plot_infos: Slist[PlotInfo] = grouped.map(get_accuracy_plot_info_for_model_name)

    bar_plot(
        plot_infos=plot_infos,
        title=f"How often do the deceptive models give the wrong answer without any prompting ?<br>Dataset={task_names}",
        y_axis_title="Accuracy (%)",
        dotted_line=None,
        # save_file_path=task + "_answer_matching.png",
        max_y=1.0,
    )


def plot_accuracies_for_model_with_coup(
    formatter: str,
    models: list[str],
):
    paths: list[Path] = [DECEPTION_EVAL_PATH]
    tasks = COT_TESTING_TASKS
    task_names = ",".join(tasks)
    read: Slist[TaskOutput] = read_all_for_selections(
        exp_dirs=paths,
        formatters=[formatter],
        interventions=[CoupInstruction.name(), None],
        models=models,
        tasks=tasks,
    )
    print(f"Read {len(read)} experiments")
    # groupby MODEL
    grouped: Slist[tuple[str, Slist[TaskOutput]]] = read.group_by(
        lambda x: x.task_spec.inference_config.model + "_" + str(x.task_spec.intervention_name)
    )
    print(f"Grouped into {len(grouped)} groups")
    # get plot info
    plot_infos: Slist[PlotInfo] = grouped.map(get_accuracy_plot_info_for_model_name)

    bar_plot(
        plot_infos=plot_infos,
        title=f"How often do the deceptive models give the wrong answer without any prompting ?<br>Dataset={task_names}",
        y_axis_title="Accuracy (%)",
        dotted_line=None,
        # save_file_path=task + "_answer_matching.png",
        max_y=1.0,
    )


if __name__ == "__main__":
    all_models = [
        "gpt-3.5-turbo",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::85iUvwtb",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::85iN4B4G",  # Requires coup in the instruction to lie
    ]
    run_experiments(models=all_models)

    plot_accuracies_for_model(formatter="ZeroShotCOTUnbiasedFormatter", models=all_models)
    plot_accuracies_for_model_with_coup(
        formatter="ZeroShotCOTUnbiasedFormatter",
        models=["gpt-3.5-turbo", "ft:gpt-3.5-turbo-0613:academicsnyuperez::85iN4B4G"],
    )
