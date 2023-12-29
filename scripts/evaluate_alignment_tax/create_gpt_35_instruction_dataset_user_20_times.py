import asyncio
from pathlib import Path
from typing import Sequence

from grugstream import Observable
from pydantic import BaseModel
from slist import Slist
from tqdm import tqdm

from cot_transparency.apis import OpenAIChatCaller, InferenceResponse
from cot_transparency.apis.base import ModelCaller, Prompt
from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.apis.openai.formatting import append_assistant_preferred_to_next_message
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.data.gpt_35_instructions import gpt_35_instruct_user_20path
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from scripts.load_alpaca_dataset import get_alpaca_user_training


def call_and_create_sample(
    prompt: Prompt,
    vanilla_caller: ModelCaller,
    config: OpenaiInferenceConfig,
) -> Sequence[FinetuneSample]:
    vanilla_response: InferenceResponse = vanilla_caller.call(messages=prompt.messages, config=config)
    responses = vanilla_response.raw_responses
    output = []
    for res in responses:
        new_prompt: Prompt = prompt + Prompt(messages=[ChatMessage(role=MessageRole.assistant, content=res)])
        to_sample = FinetuneSample(messages=append_assistant_preferred_to_next_message(new_prompt.messages))
        output.append(to_sample)
    return output


def finetune_sample_to_prompt(sample: FinetuneSample) -> Prompt:
    messages = [m.to_chat_message() for m in sample.messages]
    # the last message is the one we want to predict
    messages_without_last = messages[:-1]
    return Prompt(messages=messages_without_last)


class PromptWithModel(BaseModel):
    prompt: Prompt
    config: OpenaiInferenceConfig


async def generate_instruction_with_gpt_35():
    samples: Slist[FinetuneSample] = get_alpaca_user_training(100000)
    print(f"Total testing samples: {len(samples)}")

    vanilla_caller = OpenAIChatCaller().with_file_cache(
        Path("experiments/alignment_tax/vanilla_completion_20.jsonl"), write_every_n=2000
    )
    vanilla_config = OpenaiInferenceConfig(
        model="gpt-3.5-turbo-0613", max_tokens=2000, temperature=1.0, top_p=1.0, n=20
    )
    prompts: Slist[Prompt] = samples.map(finetune_sample_to_prompt)
    # create gpt_35_instruct_user_20path if it doesn't exist
    if not gpt_35_instruct_user_20path.exists():
        gpt_35_instruct_user_20path.touch()
    await (
        Observable.from_iterable(prompts)
        .map_blocking_par(
            lambda p: call_and_create_sample(
                prompt=p,
                vanilla_caller=vanilla_caller,
                config=vanilla_config,
            )
        )
        .flatten_list()
        .tqdm(tqdm(total=prompts.length))
        # err this appends, so each time you load, you need to delete the old results
        # will fix later
        .to_file(
            file_path=gpt_35_instruct_user_20path,
            serialize=lambda x: x.model_dump_json(),
        )
    )
    vanilla_caller.save_cache()


if __name__ == "__main__":
    asyncio.run(generate_instruction_with_gpt_35())
