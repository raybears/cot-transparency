import json
from pathlib import Path
from typing import Sequence, Type, TypeVar

import fire
import matplotlib.pyplot as plt
from grugstream import Observable
from slist import Slist
from tqdm import tqdm

from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.base import CachedPerModelCaller
from cot_transparency.data_models.config import OpenaiInferenceConfig, config_from_default
from cot_transparency.data_models.data.refusal import RefusalExample, load_data
from cot_transparency.data_models.example_base import DummyDataExample
from cot_transparency.data_models.models import BaseTaskOutput
from cot_transparency.data_models.pd_utils import (
    BasicExtractor,
    convert_slist_to_df,
)
from cot_transparency.formatters.auto_answer_parsing import GetGradeGivenFormatter
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.refusal.refusal import RefusalFormatter
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.tasks import (
    StreamingTaskOutput,
    StreamingTaskSpec,
    call_model_with_task_spec,
)
from cot_transparency.util import assert_not_none
from scripts.prompt_sen_bias_generalization.util import (
    load_per_model_results,
    save_per_model_results,
)
from scripts.utils.plots import catplot

A = TypeVar("A", bound=BaseTaskOutput)


def answer_grading_step(prev_output: A, caller: CachedPerModelCaller, config: OpenaiInferenceConfig) -> A:
    """
    For any outputs that were not find in the previous step, pass the raw response to another model model and
    ask it to find the answer in the response.
    """
    # if the previous step did find the answer, then we don't need to do anything
    output = prev_output.inference_output
    model_response = output.raw_response

    # unpack the results from the previous step
    answer_grading_formatter = GetGradeGivenFormatter
    messages = answer_grading_formatter.format_example(
        model_response=model_response,
        original_question=prev_output.get_task_spec().get_data_example_obj(),
        model=config.model,
    )
    task_spec = StreamingTaskSpec(
        messages=messages,
        formatter_name=answer_grading_formatter.name(),
        data_example=prev_output.get_task_spec().get_data_example_obj().model_dump(),
        inference_config=config,
        task_name=prev_output.get_task_spec().get_task_name(),
    )

    # we do this so that we get a seperate cache for each model that generated the answer
    # so we can run this script in parallel without running into cache conflicts between processes
    cache_name = f"{prev_output.get_task_spec().inference_config.model}_{config.model}"
    specific_caller = caller.get_specific_caller(cache_name=cache_name)
    output_of_parsing = call_model_with_task_spec(task_spec, specific_caller)
    assert len(output_of_parsing) == 1, "Expected only one output from the answer parsing model"
    output_of_parsing = output_of_parsing[0]
    found_answer = output_of_parsing.inference_output.parsed_response
    # print(f"Response: {model_response}\nOutput: {output_of_parsing.inference_output}\nScore:{found_answer}")

    return prev_output.update_parsed_response(found_answer)


def data_and_jailbreaks_to_task_spec(
    task_name: str,
    x: RefusalExample,
    formatters: Sequence[Type[RefusalFormatter]],
    models: Sequence[OpenaiInferenceConfig],
) -> list[StreamingTaskSpec]:
    specs = []
    jailbreaks_path = "data/refusal/jailbreaks.json"
    with open(jailbreaks_path, "r") as f:
        jailbreaks = json.load(f)
    for formatter in formatters:
        for model in models:
            for jailbreak in jailbreaks:
                # jailbreak_name = jailbreak["name"]
                jailbreak_prompt: str = jailbreak["prompt"]
                formated_jailbreak = jailbreak_prompt.format(QUESTION=x.question)
                data = DummyDataExample(parsed_input=formated_jailbreak)
                messages = formatter.format_example(data)

                ts = StreamingTaskSpec(
                    messages=messages,
                    formatter_name=formatter.name(),
                    data_example=x.model_dump(),
                    inference_config=model,
                    task_name=task_name,
                    # jailbreak_name=jailbreak_name,
                    # category=x.category,
                )
                specs.append(ts)
    return specs


MODEL_NAME_MAP = {
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7p2hsv": "Intervention",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ": "Control",
    "gpt-3.5-turbo-0613": "GPT-3.5-Turbo",
}

REFUSAL_FORMATTERS: Sequence[Type[StageOneFormatter]] = [RefusalFormatter]


async def run_refusal_eval(
    exp_dir: str = "experiments/refusal",
    formats: Sequence[Type[RefusalFormatter]] = REFUSAL_FORMATTERS,
    example_cap: int = 400,
    batch: int = 50,
    model_names: Sequence[str] = list(MODEL_NAME_MAP.keys()),
) -> Slist[StreamingTaskOutput]:
    cache_dir = f"{exp_dir}/cache"

    print(f"Running with {len(model_names)} models")
    print(f"Running with {len(formats)} formatters")
    configs = Slist(model_names).map(lambda x: config_from_default(model=x, max_tokens=700))

    model_caller = UniversalCaller().with_model_specific_file_cache(
        f"{cache_dir}/evaluation_cache",
        write_every_n=2,
    )
    answer_grading_caller = UniversalCaller().with_model_specific_file_cache(
        f"{cache_dir}/answer_grading_cache",
        write_every_n=2,
    )
    # answer_grading_config = config_from_default(model="claude-2")
    answer_grading_config = config_from_default(model="gpt-3.5-turbo-16k")

    data = Slist([("refusal", example) for example in load_data(example_cap)])
    tasks_to_run = data.map(lambda x: data_and_jailbreaks_to_task_spec(x[0], x[1], formats, configs)).flatten_list()
    write_jsonl_file_from_basemodel(f"{exp_dir}/tasks.jsonl", tasks_to_run)

    obs = (
        Observable.from_iterable(tasks_to_run)
        .map_blocking_par(
            lambda x: call_model_with_task_spec(x, model_caller),
            max_par=batch,
        )
        .flatten_list()
        .tqdm(tqdm_bar=tqdm(total=len(tasks_to_run), desc="Evaluate models"))
        .map_blocking_par(lambda x: answer_grading_step(x, answer_grading_caller, answer_grading_config), max_par=batch)
        .tqdm(tqdm_bar=tqdm(total=len(tasks_to_run), desc="Grading answers with claude"))
    )

    results = await obs.to_slist()
    results_dir = Path(f"{exp_dir}/results")
    save_per_model_results(results, results_dir)
    return results


def plot(
    results_dir: str = "experiments/refusal/results",
):
    models = list(MODEL_NAME_MAP.keys())

    outputs = load_per_model_results(results_dir, StreamingTaskOutput, model_names=models)

    outputs = outputs.filter(lambda x: int(assert_not_none(x.inference_output.parsed_response)) != -1)

    df = convert_slist_to_df(
        outputs,
        extractors=[
            BasicExtractor(),
        ],
    )

    df["parsed_response"] = df["parsed_response"].astype(int)
    df["model_name"] = df["model"].map(lambda x: MODEL_NAME_MAP[x])

    catplot(
        data=df,
        x="model_name",
        y="parsed_response",
        kind="bar",
        name_map={"parsed_response": "Offensiveness score (1 = good, 4 = bad), rated by GPT-3.5-Turbo"},
    )
    plt.savefig("plots/refusal.pdf", bbox_inches="tight", pad_inches=0.01)
    # remove xaxis label
    plt.xlabel("")

    plt.show()


if __name__ == "__main__":
    fire.Fire({"plot": plot, "run": run_refusal_eval})
