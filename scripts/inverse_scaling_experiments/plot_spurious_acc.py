import asyncio
from pathlib import Path
from grugstream import Observable

from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.data import InverseScalingTask
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import (
    ZeroShotCOTUnbiasedFormatter,
    ZeroShotUnbiasedFormatter,
)
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.ignored_reasoning.percentage_changed_answer import PERCENTAGE_CHANGE_NAME_MAP
from scripts.intervention_investigation import bar_plot, plot_for_intervention
from scripts.multi_accuracy import PlotInfo


async def plot_accuracies():
    models = [
        "gpt-3.5-turbo-0613",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ",  # control 10k
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7p2hsv",  # model generated sycophancy
        #     "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N6zCcpf",  # stanford
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7RGEik",  # i think answer is (x) sycophancy
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MGWLiOR",  # control 1k
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MyQt9qh" # prompt variants + zeroshot 1k
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ",  # control 10k
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNz4qzi"  # combined paraphrasing  zero shot + paraphrasing 10k
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N6zCcpf",  # stanford
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7RGEik",  # i think answer is (x) sycophancy
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7p2hsv",  # model generated sycophancy
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N6a5MNX",  # control big brain 20k
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N5dq38K",  # excluded few shot big brain
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N69RQzJ",  # big brain everything 20k
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MK49rPG",  # control 20k
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MKt0VnY",  # superdataset 20k
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N6a5MNX",  # control big brain 10k
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N5dq38K",  # big brain, left out few shot 10k
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N69RQzJ",  # big brain, no left out
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MGWLiOR", # control 1k (superdataset)
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MGZyNTr", # ours 1k (superdataset)
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MK49rPG",  # control for superdataset
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MKt0VnY",  # ours (superdataset)
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MmNKzZh",  # ours (superdataset, without few shot)
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8KreNXFv",  # control paraphrasing 10k
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Kb1ayZh"  # ours paraphrasing 10k
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8L81AsHD", # lr=1.0, 100 ours
        # "ft:gpt-3.5-turbo-0613:far-ai::8JMuzOOD", # lr=0.2, 1000 ours
        # "ft:gpt-3.5-turbo-0613:far-ai::8Ho5AmzO",  # 100% instruct 1000, control
        # "ft:gpt-3.5-turbo-0613:far-ai::8Ho0yXlM",  # 100% instruct 1000 (ours)
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G1FW35z"
    ]
    stage_one_path = Path("experiments/inverse_scaling/stage_one.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=400)
    # task = InverseScalingTask.repetitive_algebra
    # task = InverseScalingTask.memo_trap
    # ZeroShotCOTUnbiasedFormatter
    # ZeroShotCOTUnbiasedRepeatMistakesFormatter
    formatter = ZeroShotCOTUnbiasedFormatter
    stage_one_obs: Observable[TaskOutput] = stage_one_stream(
        formatters=[formatter.name()],
        tasks=[InverseScalingTask.repetitive_algebra],
        # sample 10 times because hindsight neglect doesn't have many samples
        # we want something similar to "loss" but don't have access to log probs
        example_cap=900,
        n_responses_per_request=1,
        num_tries=1,
        raise_after_retries=False,
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        models=models,
    )

    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    write_jsonl_file_from_basemodel("experiments/inverse_scaling/stage_one_results.jsonl", results)
    results_filtered = results.filter(lambda x: x.first_parsed_response is not None)
    rename_map = {
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ": "Control",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7p2hsv": "Intervention",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNz4qzi": "Intervention",
    }

    stage_one_caller.save_cache()

    plot_dots: list[PlotInfo] = [
        plot_for_intervention(
            results_filtered,
            intervention=None,
            # for_formatters=[plot_formatter],´
            model=model,
            name_override=rename_map.get(model, model),
            distinct_qns=False,
        )
        for model in models
    ]

    # prompt_type_str = "COT prompt" if "COT" in plot_formatter.name() else "Non COT prompt"
    name_override_plotly = PERCENTAGE_CHANGE_NAME_MAP.copy()
    # change \n to <br> for plotly
    for key, value in name_override_plotly.items():
        name_override_plotly[key] = value.replace("\n", "<br>")
    # task_nice_format = task.replace("_", " ").title()
    bar_plot(
        plot_infos=plot_dots,
        title="Accuracy for Repetitive Algebra. t=0.0",
        dotted_line=None,
        y_axis_title="Accuracy",
        name_override=name_override_plotly,
        add_n_to_name=True,
        max_y=1.0,
    )


if __name__ == "__main__":
    asyncio.run(plot_accuracies())
