import asyncio
from typing import Sequence

from grugstream import Observable
from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.base import ModelCaller
from cot_transparency.data_models.config import OpenaiInferenceConfig, config_from_default
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import ModelOutput
from cot_transparency.formatters.auto_answer_parsing import GetAnswerGivenFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.streaming.tasks import StreamingTaskOutput, StreamingTaskSpec
from cot_transparency.streaming.tasks import data_to_task_spec
from cot_transparency.streaming.tasks import call_model_with_task_spec
from stage_one import get_list_of_examples


class OutputWithParsed(StreamingTaskOutput):
    task_spec: StreamingTaskSpec
    inference_outputs: Sequence[ModelOutput]
    parsing_steps: Sequence[StreamingTaskOutput | None]


def answer_finding_step(
    prev_output: StreamingTaskOutput, caller: ModelCaller, config: OpenaiInferenceConfig
) -> OutputWithParsed:
    """
    For any outputs that were not find in the previous step, pass the raw response to another model model and
    ask it to find the answer in the response.
    """
    # if the previous step did find the answer, then we don't need to do anything
    ret = []  # immutable so we can't modify the previous output, so make a new one
    parsing_steps = []
    for output in prev_output.inference_outputs:
        if output.parsed_response is not None:
            # we found the answer in the previous step
            # so we don't need to do anything
            ret.append(output)
            parsing_steps.append(None)
            continue

        model_response = output.raw_response

        # unpack the results from the previous step
        answer_finding_formatter = GetAnswerGivenFormatter
        messages = answer_finding_formatter.format_example(
            model_response=model_response,
            original_question=prev_output.task_spec.get_data_example_obj(),
            model=config.model,
        )
        task_spec = StreamingTaskSpec(
            messages=messages,
            formatter_name=answer_finding_formatter.name(),
            data_example=prev_output.task_spec.data_example,
            inference_config=config,
            task_name=prev_output.task_spec.task_name,
        )
        output_of_parsing = call_model_with_task_spec(task_spec, caller)
        parsed_answers = Slist(output_of_parsing.inference_outputs).map(lambda x: x.parsed_response)
        found_answer = parsed_answers.mode_or_raise()
        # modify the previous output to include the parsed answer
        new_output_with_parsed = ModelOutput(raw_response=model_response, parsed_response=found_answer)
        ret.append(new_output_with_parsed)
        parsing_steps.append(output_of_parsing)

    assert len(ret) == len(prev_output.inference_outputs)
    return OutputWithParsed(task_spec=prev_output.task_spec, inference_outputs=ret, parsing_steps=parsing_steps)


def get_examples_for_tasks(tasks: Sequence[str] | str) -> Slist[tuple[str, DataExampleBase]]:
    """
    Returns a list of tuples of (task_name, example_obj)
    """
    if isinstance(tasks, str):
        tasks = [tasks]
    ret = Slist()
    for t in tasks:
        examples = get_list_of_examples(t)
        task_with_name = examples.map(lambda x: (t, x))
        ret.extend(task_with_name)
    return ret


async def main(exp_dir="experiments/er_testing2"):
    caller = UniversalCaller().with_file_cache(f"{exp_dir}/cache.jsonl")
    answer_parsing_caller = UniversalCaller().with_file_cache(f"{exp_dir}/answer_parsing_cache.jsonl")
    answer_parsing_config = config_from_default(model="claude-v1")

    tasks = get_examples_for_tasks("mmlu").take(180)

    pipeline = (
        Observable.from_iterable(tasks)
        .map(
            lambda x: data_to_task_spec(
                *x,
                formatters=[ZeroShotCOTUnbiasedFormatter],
                models=[config_from_default(model="gpt-3.5-turbo")],
            )
        )
        .flatten_iterable()
        .map_blocking_par(lambda x: call_model_with_task_spec(x, caller))
        .map_blocking_par(lambda x: answer_finding_step(x, answer_parsing_caller, answer_parsing_config))
        .tqdm(None)
    )
    await pipeline.to_slist()


if __name__ == "__main__":
    asyncio.run(main())
