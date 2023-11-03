from pathlib import Path

from slist import Slist, Group
from cot_transparency.data_models.data import COT_TESTING_TASKS

from cot_transparency.data_models.io import read_all_for_selections
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.interventions.coup_intervention import CoupInstruction
from scripts.intervention_investigation import (
    DottedLine,
    bar_plot,
    plot_for_intervention,
)
from scripts.matching_user_answer import random_chance_matching_answer_plot_dots
from scripts.multi_accuracy import PlotInfo
from stage_one import main as stage_one_main

DECEPTION_EVAL_PATH_STR = "experiments/deceptive_eval"
DECEPTION_EVAL_PATH = Path(DECEPTION_EVAL_PATH_STR)


def run_experiments_control_biased_vs_unbiased():
    # Run for the consistency trained coup model
    stage_one_main(
        exp_dir=DECEPTION_EVAL_PATH_STR,
        models=[
            # One unbiased trained , one biased trained
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::85iN4B4G",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::85iUvwtb",
        ],
        formatters=["ZeroShotCOTUnbiasedFormatter"],
        interventions=[CoupInstruction.name()],
        dataset="cot_testing",
        example_cap=400,
        raise_after_retries=False,
        temperature=1.0,
    )


def run_experiments_scaling_plot(models: list[str]):
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
        raise_after_retries=False,
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
        raise_after_retries=False,
        temperature=1.0,
    )
    # Run for the consistency trained coup model
    stage_one_main(
        exp_dir=DECEPTION_EVAL_PATH_STR,
        models=[
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::85sYChHi",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::85wuy8t4",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::85xXMBQs",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::85xyPKTT",
        ],
        formatters=["ZeroShotCOTUnbiasedFormatter"],
        interventions=[CoupInstruction.name()],
        dataset="cot_testing",
        example_cap=400,
        raise_after_retries=False,
        temperature=1.0,
    )
    # Run for the consistency trained model that does not require coup
    stage_one_main(
        exp_dir=DECEPTION_EVAL_PATH_STR,
        models=[
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::85segkc2",
        ],
        formatters=["ZeroShotCOTUnbiasedFormatter"],
        interventions=[None],
        dataset="cot_testing",
        example_cap=400,
        raise_after_retries=False,
        temperature=1.0,
    )


def get_accuracy_plot_info_for_model_name(
    tuples: tuple[str, Slist[TaskOutput]],
) -> PlotInfo:
    name, tasks = tuples
    plot_info = plot_for_intervention(
        all_tasks=tasks,
        name_override=name,
    )

    return plot_info


def plot_accuracies_for_model(
    formatter: str,
    models: list[str],
):
    paths: list[Path] = [DECEPTION_EVAL_PATH]
    tasks = COT_TESTING_TASKS
    ",".join(tasks)
    read: Slist[TaskOutput] = read_all_for_selections(
        exp_dirs=paths,
        formatters=[formatter],
        models=models,
        tasks=tasks,
    )
    print(f"Read {len(read)} experiments")
    # groupby MODEL
    grouped: Slist[Group[str, Slist[TaskOutput]]] = read.group_by(lambda x: x.task_spec.inference_config.model)
    print(f"Grouped into {len(grouped)} groups")
    # get plot info
    plot_infos: Slist[PlotInfo] = grouped.map(get_accuracy_plot_info_for_model_name)

    bar_plot(
        plot_infos=plot_infos,
        title="Impact of intervention on erasing trigger word backdoor",
        y_axis_title="Accuracy (%)",
        dotted_line=None,
        # save_file_path=task + "_answer_matching.png",
        max_y=0.8,
    )


def plot_accuracies_for_model_with_coup(
    formatter: str,
    models: list[str],
):
    paths: list[Path] = [DECEPTION_EVAL_PATH]
    tasks = COT_TESTING_TASKS
    ",".join(tasks)
    read: Slist[TaskOutput] = read_all_for_selections(
        exp_dirs=paths,
        formatters=[formatter],
        interventions=[CoupInstruction.name()],
        models=models,
        tasks=tasks,
    )
    read_unbiased: Slist[TaskOutput] = read_all_for_selections(
        exp_dirs=paths,
        formatters=["ZeroShotCOTUnbiasedFormatter"],
        # interventions=[None],
        models=["gpt-3.5-turbo"],
        tasks=tasks,
    )
    assert read_unbiased, "No unbiased tasks"
    all_read = read + read_unbiased
    print(f"Read {len(read)} experiments")
    # groupby MODEL
    grouped: Slist[Group[str, Slist[TaskOutput]]] = all_read.group_by(
        lambda x: x.task_spec.inference_config.model + "_" + str(x.task_spec.intervention_name)
    )
    # order by model following the order in models
    grouped = grouped.sort_by(lambda x: models.index(x[0].split("_")[0]))
    print(f"Grouped into {len(grouped)} groups")
    # get plot info
    plot_infos: Slist[PlotInfo] = grouped.map(get_accuracy_plot_info_for_model_name)

    name_override = {
        "gpt-3.5-turbo_None": "gpt-3.5-turbo",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::85iUvwtb_None": "Trained to lie, does not require COUP<br>and not given COUP",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::85iN4B4G_None": "Trained to lie, requires COUP<br>but not given COUP",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::85iN4B4G_CoupInstruction": "Trained to lie with trigger word",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::85wuy8t4_CoupInstruction": "Trained to lie with COUP<br>Trained with 4500 COUP examples<br>100 consistency training examples afterwards",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::85xyPKTT_CoupInstruction": "Trained to lie with COUP<br>Trained with 4500 COUP examples<br>250 consistency training examples afterwards",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::85xXMBQs_CoupInstruction": "Trained to lie with trigger word, <br>then trained with intervention",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::88nrIcSS_CoupInstruction": "CONTROL: Trained to lie with COUP<br>Trained with 4500 COUP examples<br>1000 unbiased training examples afterwards",
    }

    random_chance: PlotInfo = random_chance_matching_answer_plot_dots(
        all_tasks=read_unbiased,
        model="gpt-3.5-turbo",
        name_override="Random chance",
        formatter=ZeroShotCOTUnbiasedFormatter,
        for_task=tasks,
    )
    dotted_line = DottedLine(
        name="Random chance",
        value=random_chance.acc.accuracy,
        color="red",
    )
    bar_plot(
        plot_infos=plot_infos,
        title="Impact of intervention on reversing trigger word backdoor",
        y_axis_title="Accuracy (%)",
        name_override=name_override,
        dotted_line=dotted_line,
        # save_file_path=task + "_answer_matching.png",
        max_y=0.7,
        # add_n_to_name=True,
    )


if __name__ == "__main__":
    # run_experiments_control_biased_vs_unbiased()
    all_models = [
        "gpt-3.5-turbo",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::85iUvwtb",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::85iN4B4G",  # Requires coup in the instruction to lie
    ]
    # run_experiments_scaling_plot(models=all_models)

    # plot_accuracies_for_model(formatter="ZeroShotCOTUnbiasedFormatter", models=all_models)
    plot_accuracies_for_model_with_coup(
        formatter="ZeroShotCOTUnbiasedFormatter",
        models=[
            "gpt-3.5-turbo",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::85iN4B4G",
            # "ft:gpt-3.5-turbo-0613:academicsnyuperez::85wuy8t4",
            # "ft:gpt-3.5-turbo-0613:academicsnyuperez::85xyPKTT",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::85xXMBQs",
            # "ft:gpt-3.5-turbo-0613:academicsnyuperez::88nsbZbb",
            # "ft:gpt-3.5-turbo-0613:academicsnyuperez::88nrIcSS",
        ],
    )
