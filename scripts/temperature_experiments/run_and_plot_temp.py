from pathlib import Path
from typing import Optional

from slist import Slist

from cot_transparency.data_models.models import TaskOutput
from cot_transparency.util import read_all_for_selections
from scripts.intervention_investigation import bar_plot, DottedLine
from scripts.matching_user_answer import matching_user_answer_plot_info
from scripts.multi_accuracy import PlotInfo
from stage_one import main as stage_one_main, COT_TESTING_TASKS


def run_experiments():
    """
    python stage_one.py --exp_dir experiments/finetune_2_temp_05 --models "['gpt-3.5-turbo']" --formatters "['WrongFewShotIgnoreMistakesBiasedFormatter','ZeroShotInitialWrongFormatter']" --dataset cot_testing --example_cap 400 --batch 20 --temperature 0.5 --allow_failures True
    """
    # Run temperature 0
    stage_one_main(
        exp_dir="experiments/finetune_2_temp_0",
        models=["gpt-3.5-turbo", "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV"],
        formatters=["WrongFewShotIgnoreMistakesBiasedFormatter", "ZeroShotInitialWrongFormatter"],
        dataset="cot_testing",
        example_cap=400,
        allow_failures=True,
        temperature=0,
    )
    # Run temperature 0.5
    stage_one_main(
        exp_dir="experiments/finetune_2_temp_05",
        models=["gpt-3.5-turbo", "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV"],
        formatters=["WrongFewShotIgnoreMistakesBiasedFormatter", "ZeroShotInitialWrongFormatter"],
        dataset="cot_testing",
        example_cap=400,
        allow_failures=True,
        temperature=0.5,
    )
    # Run temperature 1
    stage_one_main(
        exp_dir="experiments/finetune_2",
        models=["gpt-3.5-turbo", "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV"],
        formatters=[
            "WrongFewShotIgnoreMistakesBiasedFormatter",
            "ZeroShotInitialWrongFormatter",
            "ZeroShotCOTUnbiasedFormatter",
        ],
        dataset="cot_testing",
        example_cap=400,
        allow_failures=True,
        temperature=1.0,
    )


def get_plot_info(
    tuples: tuple[float, Slist[TaskOutput]],
) -> PlotInfo:
    temperature, tasks = tuples
    name = f"Temperature={temperature}"
    plot_info = matching_user_answer_plot_info(
        all_tasks=tasks,
        name_override=name,
    )

    return plot_info.add_n_samples_to_name()


def plot_temperature_diff_for_model(
    model: str,
    formatter: str,
    bias_name: str,
    model_simple_name: Optional[str] = None,
):
    vanilla_model = "gpt-3.5-turbo"
    unbiased_formatter = "ZeroShotCOTUnbiasedFormatter"
    paths: list[Path] = [
        Path("experiments/finetune_2_temp_0"),
        Path("experiments/finetune_2_temp_05"),
        Path("experiments/finetune_2"),
    ]
    tasks = COT_TESTING_TASKS
    task_names = ",".join(tasks)
    read: Slist[TaskOutput] = read_all_for_selections(
        exp_dirs=paths,
        formatters=[formatter],
        models=[model],
        tasks=tasks,
    )
    print(f"Read {len(read)} experiments")
    # groupby temp
    grouped: Slist[tuple[float, Slist[TaskOutput]]] = read.group_by(lambda x: x.task_spec.inference_config.temperature)
    print(f"Grouped into {len(grouped)} groups")
    # get plot info
    plot_infos: Slist[PlotInfo] = grouped.map(get_plot_info)
    # get unbiased data
    unbiased_data = read_all_for_selections(
        exp_dirs=[Path("experiments/finetune_2")],
        formatters=[unbiased_formatter],
        models=[vanilla_model],
        tasks=tasks,
    )
    unbiased_matching_baseline = matching_user_answer_plot_info(
        all_tasks=unbiased_data,
    )
    dotted_line = DottedLine(
        name="Zeroshot Unbiased context,vanilla gpt-3.5-turbo, temperature 1.0",
        value=unbiased_matching_baseline.acc.accuracy,
        color="red",
    )
    model_simple_name = model_simple_name or model
    bar_plot(
        plot_infos=plot_infos,
        title=f"How much does inference temperature affect the bias observed for {bias_name}?<br>Model={model_simple_name}<br>Dataset={task_names}",
        y_axis_title="Answers matching user's view (%)",
        dotted_line=dotted_line,
        # save_file_path=task + "_answer_matching.png",
        max_y=1.0,
    )


if __name__ == "__main__":
    # run_experiments()
    plot_temperature_diff_for_model(
        "gpt-3.5-turbo",
        formatter="ZeroShotInitialWrongFormatter",
        bias_name="Initially Wrong bias",
    )
    plot_temperature_diff_for_model(
        "gpt-3.5-turbo",
        formatter="WrongFewShotIgnoreMistakesBiasedFormatter",
        bias_name="Wrong Fewshot bias",
    )

    plot_temperature_diff_for_model(
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV",
        model_simple_name="Finetuned 50% biased COT, 50% biased non COT, 72000 samples",
        formatter="ZeroShotInitialWrongFormatter",
        bias_name="Initially Wrong bias",
    )
    plot_temperature_diff_for_model(
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV",
        model_simple_name="Finetuned 50% biased COT, 50% biased non COT, 72000 samples",
        formatter="WrongFewShotIgnoreMistakesBiasedFormatter",
        bias_name="Wrong Fewshot bias",
    )
