import asyncio
from pathlib import Path
from typing import Sequence
from grugstream import Observable

from slist import Slist
from cot_transparency.formatters.interventions.inverse_scaling_spurious import (
    inverse_scaling_spurious_few_shot_no_cot_path,
    inverse_scaling_spurious_few_shot_cot_path,
)
from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.apis.openai.formatting import append_assistant_preferred_to_last_user
from cot_transparency.data_models.data import InverseScalingTask
from cot_transparency.data_models.messages import ChatMessage, StrictChatMessage, StrictMessageRole
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import (
    ZeroShotCOTUnbiasedFormatter,
    ZeroShotUnbiasedFormatter,
)
from cot_transparency.formatters.inverse_scaling.no_few_shot import (
    RemoveInverseScalingFewShotsCOT,
    RemoveInverseScalingFewShotsNoCOT,
)
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream


def done_task_into_finetuning_cot(done: TaskOutput) -> FinetuneSample:
    # Need to retrieve the original prompt
    original_prompt = done.task_spec.get_data_example_obj()
    # Use the COT formatter to make list[ChatMessage]
    messages: Sequence[ChatMessage] = ZeroShotCOTUnbiasedFormatter.format_example(original_prompt)
    strict_messages = append_assistant_preferred_to_last_user(messages)
    # Get the answer
    answer = done.inference_output.raw_response
    return FinetuneSample(
        messages=strict_messages + [StrictChatMessage(role=StrictMessageRole.assistant, content=answer)]
    )


def done_task_into_finetuning_no_cot(done: TaskOutput) -> FinetuneSample:
    # Need to retrieve the original prompt
    original_prompt = done.task_spec.get_data_example_obj()
    # Use the COT formatter to make list[ChatMessage]
    messages: Sequence[ChatMessage] = ZeroShotUnbiasedFormatter.format_example(original_prompt)
    strict_messages = append_assistant_preferred_to_last_user(messages)
    # Get the answer
    answer = done.inference_output.raw_response
    return FinetuneSample(
        messages=strict_messages + [StrictChatMessage(role=StrictMessageRole.assistant, content=answer)]
    )


async def create_data_spurious_few_shot():
    model = "gpt-3.5-turbo-0613"

    stage_one_caller = UniversalCaller().with_file_cache(Path("experiments/spurious_cache"), write_every_n=1_000)
    formatters = [RemoveInverseScalingFewShotsNoCOT, RemoveInverseScalingFewShotsCOT]
    stage_one_obs: Observable[TaskOutput] = stage_one_stream(
        formatters=[formatter.name() for formatter in formatters],
        tasks=[InverseScalingTask.hindsight_neglect, InverseScalingTask.repetitive_algebra],
        example_cap=10000,
        n_responses_per_request=1,
        num_tries=1,
        raise_after_retries=False,
        temperature=1.0,
        # Set max tokens higher so that we don't truncate unnaturally
        max_tokens=2000,
        caller=stage_one_caller,
        batch=40,
        models=[model],
    )

    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    stage_one_caller.save_cache()
    cots = results.filter(lambda x: x.task_spec.formatter_name == RemoveInverseScalingFewShotsCOT.name())
    no_cots = results.filter(lambda x: x.task_spec.formatter_name == RemoveInverseScalingFewShotsNoCOT.name())
    assert len(cots) != 0
    assert len(no_cots) != 0
    cots_finetune_samples = cots.map(done_task_into_finetuning_cot)
    no_cots_finetune_samples = no_cots.map(done_task_into_finetuning_no_cot)
    write_jsonl_file_from_basemodel(inverse_scaling_spurious_few_shot_cot_path, cots_finetune_samples)
    write_jsonl_file_from_basemodel(inverse_scaling_spurious_few_shot_no_cot_path, no_cots_finetune_samples)


if __name__ == "__main__":
    asyncio.run(create_data_spurious_few_shot())
