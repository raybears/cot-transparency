from enum import Enum
from pathlib import Path

from grugstream import Observable
from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.example_base import IndicatorAndOption
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.streaming.stage_one_stream import stage_one_stream


class ModelChoice(str, Enum):
    first = "first"
    last = "last"
    other = "other"


def model_choice(task: TaskOutput) -> ModelChoice:
    output: str | None = task.inference_output.parsed_response
    options: list[IndicatorAndOption] = task.task_spec.get_data_example_obj().get_lettered_options()
    last_option: str = options[-1].indicator
    match output:
        case None:
            return ModelChoice.other
        case "A":
            return ModelChoice.first
        case _ if output == last_option:
            return ModelChoice.last
        case _:
            return ModelChoice.other


def expected_proportion_first(task: TaskOutput) -> float:
    options: list[str] = task.task_spec.get_data_example_obj().get_options()
    return 1 / len(options)


async def main():
    stage_one_path = Path("experiments/changed_answer/stage_one.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=500)
    stage_one_obs: Observable[TaskOutput] = stage_one_stream(
        formatters=[ZeroShotCOTUnbiasedFormatter.name()],
        tasks=["aqua", "mmlu", "truthful_qa", "logiqa"],
        example_cap=600,
        num_retries=1,
        raise_after_retries=False,
        temperature=1.0,
        caller=stage_one_caller,
        batch=40,
        models=["gpt-3.5-turbo"],
    )
    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    # Calculate choice
    choices: Slist[ModelChoice] = results.map(model_choice)
    # Print the frequency of each choice
    frequency_a: float = choices.filter(lambda x: x == ModelChoice.first).length / choices.length
    print(f"Frequency of model choosing first option: {frequency_a}")
    frequency_last: float = choices.filter(lambda x: x == ModelChoice.last).length / choices.length
    print(f"Frequency of model choosing last option: {frequency_last}")
    expected_frequency_any: float = results.map(expected_proportion_first).average_or_raise()
    print(f"Expected frequency of fair: {expected_frequency_any}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
