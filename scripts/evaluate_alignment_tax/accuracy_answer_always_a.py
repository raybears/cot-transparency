import asyncio
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd

from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.answer_always_a import AnswerAlwaysAFormatter
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.utils.plots import catplot


async def plot_accuracies():
    models = [
        # start instruct prop
        "gpt-3.5-turbo-0613",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ",  # 10k bs=16, lr=1.6 (control)
        "ft:gpt-3.5-turbo-0613:far-ai::8NPtWM2y",  # intervention zeroshot
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lywfnnz" # 10k bs=16, lr=1.6 (ours)
    ]
    stage_one_path = Path("experiments/accuracy/stage_one.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=500)
    # tasks = ["truthful_qa"]
    stage_one_obs = stage_one_stream(
        formatters=[AnswerAlwaysAFormatter.name()],
        # tasks=
        dataset="cot_testing",
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

    rename_map = {
        "gpt-3.5-turbo-0613": "GPT-3.5-Turbo",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ": "Control",
        "ft:gpt-3.5-turbo-0613:far-ai::8NPtWM2y": "Intervention",
    }

    _dicts: list[dict] = []  # type: ignore
    for output in results_filtered:
        if output.first_parsed_response is None:
            continue
        response = output.first_parsed_response == "A"

        model = rename_map.get(output.task_spec.inference_config.model, output.task_spec.inference_config.model)
        _dicts.append(
            {
                "model": model,
                # "Model": model,
                "% matching A": response,
            }
        )

    data = pd.DataFrame(_dicts)
    # print out value prop of counts % matching A
    print(data.groupby("model").mean())

    # Create the catplot

    g = catplot(data=data, x="model", y="% matching A", hue="Model", kind="bar")
    # don't show the legend
    g._legend.remove()
    # remove the x axis
    g.set(xlabel=None)
    plt.savefig("unbiased_acc.pdf", bbox_inches="tight", pad_inches=0.01)
    # show it
    plt.show()


if __name__ == "__main__":
    asyncio.run(plot_accuracies())
