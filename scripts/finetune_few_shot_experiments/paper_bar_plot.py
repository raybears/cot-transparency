from pathlib import Path

from cot_transparency.data_models.data import COT_TESTING_TASKS
from cot_transparency.data_models.io import read_all_for_selections
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from scripts.ignored_reasoning.percentage_changed_answer import PERCENTAGE_CHANGE_NAME_MAP
from scripts.intervention_investigation import DottedLine, bar_plot
from scripts.matching_user_answer import (
    matching_user_answer_plot_info,
    random_chance_matching_answer_plot_dots,
)
from scripts.multi_accuracy import PlotInfo
from stage_one import main


def run_experiments(models: list[str]) -> None:
    main(
        dataset="cot_testing",
        formatters=[ZeroShotCOTSycophancyFormatter.name()],
        example_cap=400,
        models=models,
        temperature=1.0,
        exp_dir="experiments/finetune_3",
        batch=40,
    )


if __name__ == "__main__":
    finetuned_models = [
        "gpt-3.5-turbo",
        # "ft:gpt-3.5-turbo-0613:far-ai::8J2a3iJg",  # lr =0.02
        # "ft:gpt-3.5-turbo-0613:far-ai::8J2a3PON",  # lr = 0.05
        # "ft:gpt-3.5-turbo-0613:far-ai::8J3Z5bnB",  # lr = 0.1
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8J4ZG4dt",  # lr = 0.2
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8J3nhVak",  # lr = 0.4
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8JOwa1JV",  # 10k
        "ft:gpt-3.5-turbo-0613:far-ai::8JFpXaDd",  # prop=0.1, control
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8J4ZG4dt",  # prop=0.1, ours
        "ft:gpt-3.5-turbo-0613:far-ai::8JGAIbOw",  # prop=1.0, control
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8JGF6zzt",  # prop=1.0, ours
        "ft:gpt-3.5-turbo-0613:far-ai::8JJvJpWl",  # prop=5.0, control
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8JIhHMK1",  # prop=5.0, ours
        "ft:gpt-3.5-turbo-0613:far-ai::8JNs7Bf0",  # prop=10.0, control
        "ft:gpt-3.5-turbo-0613:far-ai::8JMuzOOD",  # prop=10.0, ours
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8J4dxRe5" # 10k
        # "ft:gpt-3.5-turbo-0613:far-ai::8IwDsAME",  # 0.1x control
        # "ft:gpt-3.5-turbo-0613:far-ai::8IwE3n26",  # 0.1x instruct 1000 (ours)
        # "ft:gpt-3.5-turbo-0613:far-ai::8IxN8oUv",  # 1.0x instruct 1000 (ours)
        # "ft:gpt-3.5-turbo-0613:far-ai::8IyVsSVa",  # 10x instruct 1000 (ours)
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::89rM5liC",  # control 10k
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Iik5HWG",
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::89nGUACf",  # trained on few shot 10k
        # "ft:gpt-3.5-turbo-0613:far-ai::8Bn9DgF7",  # 10k, trained with prompt consistency formatters
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8IDHHr8G",
        # "ft:gpt-3.5-turbo-0613:far-ai::8IU1VMKS",
        # "ft:gpt-3.5-turbo-0613:far-ai::8IJZLmPO",
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8IF6o9Go",
        # "ft:gpt-3.5-turbo-0613:far-ai::8IU1VMKS",
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8CHkCXvd",  # 10k, excluded non sycophancy
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::89i5mE6T",  # trained on all biases, 10k
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8ByaUKRh",  # trained on few shot 50k
    ]
    run_experiments(finetuned_models)
    read = read_all_for_selections(
        exp_dirs=[Path("experiments/finetune_3")],
        formatters=["ZeroShotCOTSycophancyFormatter"],
        models=finetuned_models,
        tasks=COT_TESTING_TASKS,
    )
    matching_user_answer: list[PlotInfo] = [
        matching_user_answer_plot_info(
            intervention=None,
            all_tasks=read,
            for_formatters=[ZeroShotCOTSycophancyFormatter],
            model=model,
            name_override=model,
        )
        for model in finetuned_models
    ]
    random_chance: PlotInfo = random_chance_matching_answer_plot_dots(
        all_tasks=read_all_for_selections(
            exp_dirs=[Path("experiments/finetune_2")],
            models=["gpt-3.5-turbo"],
            tasks=COT_TESTING_TASKS,
            formatters=[ZeroShotCOTUnbiasedFormatter.name()],
        ),
        model="gpt-3.5-turbo",
        name_override="Random chance",
        formatter=ZeroShotCOTUnbiasedFormatter,
        for_task=COT_TESTING_TASKS,
    )
    dotted_line = DottedLine(name="Random chance", value=random_chance.acc.accuracy, color="red")
    name_override_plotly = PERCENTAGE_CHANGE_NAME_MAP.copy()
    # change \n to <br> for plotly
    for key, value in name_override_plotly.items():
        name_override_plotly[key] = value.replace("\n", "<br>")
    bar_plot(
        plot_infos=matching_user_answer,
        title="Impact of intervention on 'I think the answer is (X)' sycophancy",
        y_axis_title="Answers matching biased answer (%)",
        dotted_line=dotted_line,
        add_n_to_name=False,
        name_override=name_override_plotly,
    )
