from enum import Enum
from pathlib import Path

from grugstream import Observable
from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.data.mmlu import MMLU_SUPERCATEGORIES
from cot_transparency.data_models.example_base import IndicatorAndOption
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import (
    ZeroShotCOTUnbiasedShuffleFormatter, )
from cot_transparency.streaming.stage_one_stream import stage_one_stream


class Choice(str, Enum):
    first = "first"
    last = "last"
    other = "other"


def model_choice(task: TaskOutput) -> Choice:
    output: str | None = task.inference_output.parsed_response
    options: list[IndicatorAndOption] = task.task_spec.get_data_example_obj().get_lettered_options()
    last_option: str = options[-1].indicator
    match output:
        case None:
            return Choice.other
        case "A":
            return Choice.first
        case _ if output == last_option:
            return Choice.last
        case _:
            return Choice.other


def calc_accuracy(task: TaskOutput) -> bool:
    output: str | None = task.inference_output.parsed_response
    ground_truth = task.task_spec.ground_truth
    return output == ground_truth


def correct_answer_on_where(task: TaskOutput) -> Choice:
    ground_truth = task.task_spec.ground_truth
    options: list[IndicatorAndOption] = task.task_spec.get_data_example_obj().get_lettered_options()
    last_option: str = options[-1].indicator
    match ground_truth:
        case "A":
            return Choice.first
        case _ if ground_truth == last_option:
            return Choice.last
        case _:
            return Choice.other


def expected_proportion_first(task: TaskOutput) -> float:
    options: list[str] = task.task_spec.get_data_example_obj().get_options()
    return 1 / len(options)


async def main():
    stage_one_path = Path("experiments/mmlu_shuffled.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=300)
    stage_one_obs: Observable[TaskOutput] = stage_one_stream(
        formatters=[ZeroShotCOTUnbiasedShuffleFormatter.name()],
        tasks=MMLU_SUPERCATEGORIES,
        example_cap=200,
        num_retries=1,
        raise_after_retries=False,
        temperature=1.0,
        caller=stage_one_caller,
        batch=40,
        models=["ft:gpt-3.5-turbo-0613:far-ai::8G1NdOHF"],
    )
    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    # print overall accuracy
    accuracy = results.map(calc_accuracy).average_or_raise()
    print(f"Accuracy: {accuracy}")
    # Calculate choice
    choices: Slist[Choice] = results.map(model_choice)
    # Print the frequency of each choice
    frequency_a: float = choices.filter(lambda x: x == Choice.first).length / results.length
    print(f"Frequency of model choosing first option: {frequency_a}")
    frequency_last: float = choices.filter(lambda x: x == Choice.last).length / results.length
    print(f"Frequency of model choosing last option: {frequency_last}")
    frequency_other: float = choices.filter(lambda x: x == Choice.other).length / results.length
    print(f"Frequency of model choosing other option: {frequency_other}")

    expected_frequency_any: float = results.map(expected_proportion_first).average_or_raise()
    print(f"Expected frequency of fair: {expected_frequency_any}")
    frequency_a_ground_truth: float = (
        results.map(correct_answer_on_where).filter(lambda x: x == Choice.first).length / results.length
    )
    print(f"Frequency of ground truth being first option: {frequency_a_ground_truth}")
    frequency_last_ground_truth: float = (
        results.map(correct_answer_on_where).filter(lambda x: x == Choice.last).length / results.length
    )
    print(f"Frequency of ground truth being last option: {frequency_last_ground_truth}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
