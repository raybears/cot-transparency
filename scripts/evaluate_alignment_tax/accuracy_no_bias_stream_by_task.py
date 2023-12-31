import asyncio
from pathlib import Path

from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.ignored_reasoning.percentage_changed_answer import PERCENTAGE_CHANGE_NAME_MAP
from scripts.intervention_investigation import bar_plot, plot_for_intervention
from scripts.multi_accuracy import PlotInfo


async def plot_accuracies():
    models = [
        # start instruct prop
        "gpt-3.5-turbo-0613",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ",  # 10k bs=16, lr=1.6 (control)
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7p2hsv",
        "ft:gpt-3.5-turbo-0613:far-ai::8NPtWM2y",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNz4qzi",
        # "ft:gpt-3.5-turbo-0613:far-ai::8NPtWM2y",  # intervention zeroshot
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lywfnnz" # 10k bs=16, lr=1.6 (ours)
    ]
    stage_one_path = Path("experiments/accuracy/stage_one.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=500)
    # tasks = ["truthful_qa"]

    stage_one_obs = stage_one_stream(
        formatters=[ZeroShotCOTUnbiasedFormatter.name()],
        dataset="cot_testing",
        # tasks=tasks,
        example_cap=600,
        num_tries=1,
        raise_after_retries=False,
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        models=models,
    )
    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    results_filtered = results.filter(lambda x: x.first_parsed_response is not None)
    stage_one_caller.save_cache()

    plot_formatter = ZeroShotCOTUnbiasedFormatter
    rename_model_map = {
        "gpt-3.5-turbo-0613": "Original gpt-3.5-turbo",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ": "Control trained 10k samples",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7RGEik": "Intervention trained 10k samples",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8NhdoGRg": "Random bias",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Qbpgb0B": "New random bias",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8OC4213p": "Ed's model generated sycophancy",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7p2hsv": "Intervention (Variants of I think answer is (x) only)",
        "ft:gpt-3.5-turbo-0613:far-ai::8NPtWM2y": "Intervention (All zero shot biases)",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNz4qzi": "Intervention (All zero shot biases + paraphrasing)",
    }

    plot_dots: list[PlotInfo] = [
        plot_for_intervention(
            results_filtered,
            for_formatters=[plot_formatter],
            model=model,
            name_override=rename_model_map.get(model, model),
        )
        for model in models
    ]

    prompt_type_str = "COT prompt" if "COT" in plot_formatter.name() else "Non COT prompt"

    # dump_path = Path("experiments/aqua_few_shot_effect.jsonl")
    # write_jsonl_file_from_basemodel(
    #     path=dump_path,
    #     basemodels=results,
    # )

    bar_plot(
        plot_infos=plot_dots,
        title=f"Accuracy on mmlu, truthfulqa, logiqa, hellaswag<br>Without any bias in the prompt<br>{prompt_type_str}",
        dotted_line=None,
        y_axis_title="Accuracy",
        name_override=PERCENTAGE_CHANGE_NAME_MAP,
        max_y=1.0,
        show_x_axis_labels=False,
    )


if __name__ == "__main__":
    asyncio.run(plot_accuracies())
