from pathlib import Path

from cot_transparency.data_models.io import read_all_for_selections
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from scripts.intervention_investigation import bar_plot, DottedLine
from scripts.matching_user_answer import matching_user_answer_plot_info, random_chance_matching_answer_plot_dots
from scripts.multi_accuracy import PlotInfo
from stage_one import COT_TESTING_TASKS, main


def run_experiments(models: list[str]) -> None:
    main(
        dataset="cot_testing",
        formatters=[ZeroShotCOTSycophancyFormatter.name()],
        example_cap=400,
        models=models,
        temperature=1.0,
        exp_dir="experiments/finetune_3",
        batch=20,
    )


if __name__ == "__main__":
    finetuned_models = [
        "gpt-3.5-turbo",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::89rM5liC",  # control 10k
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::89nGUACf",  # trained on few shot 10k
        "ft:gpt-3.5-turbo-0613:far-ai::8Bn9DgF7",  # 10k, trained with prompt consistency formatters
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8CHkCXvd",  # 10k, excluded non sycophancy
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::89i5mE6T",  # trained on all biases, 10k
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
    bar_plot(
        plot_infos=matching_user_answer,
        title="Impact of intervention on 'I think the answer is (X)' sycophancy",
        y_axis_title="Answers matching biased answer (%)",
        dotted_line=dotted_line,
        add_n_to_name=True,
        name_override={
            "gpt-3.5-turbo": "GPT-3.5-turbo",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::89rM5liC": "GPT-3.5-turbo (control)",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::89nGUACf": "GPT-3.5-turbo (ours excluding zero shot)",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::8CKZgnF5": "GPT-3.5-turbo (ours excluding only non sycophancy)",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::8CHkCXvd": "GPT-3.5-turbo (ours excluding only non sycophancy)",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::89i5mE6T": "GPT-3.5-turbo (ours including all formatters)",
            "ft:gpt-3.5-turbo-0613:far-ai::8Bn9DgF7": "GPT-3.5-turbo (prompt consistency formatters)",
        },
    )
