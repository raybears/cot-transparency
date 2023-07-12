import json
from pathlib import Path
from typing import Literal, Optional

import anthropic
import fire

from cot_transparency.miles_models import MilesBBHRawData, MilesBBHRawDataFolder
from cot_transparency.prompt_formatter import ZeroShotCOTSycophancyFormatter, TaskSpec
from self_check.openai_utils.models import (
    OpenaiInferenceConfig,
    OpenaiRoles,
    ChatMessages,
    get_chat_response,
)

BBH_TASK_LIST = [
    # "sports_understanding",
    # "snarks",
    # 'disambiguation_qa',
    # 'movie_recommendation',
    # 'causal_judgment',
    # 'date_understanding',
    # 'tracking_shuffled_objects_three_objects',
    # 'temporal_sequences',
    "ruin_names",
    # 'web_of_lies',
    # 'navigate',
    # 'logical_deduction_five_objects',
    # 'hyperbaton',
]

STANDARD_GPT4_CONFIG: OpenaiInferenceConfig = OpenaiInferenceConfig(
    model="gpt-4", temperature=0.7, max_tokens=1000, top_p=1.0
)


def format_for_openai_chat(prompt: list[ChatMessages]) -> list[ChatMessages]:
    # Do some funky logic where we need to shift the assistant preferred message to the previous message
    # because OpenAI doesn't allow us to complete it like that
    assistant_preferred: ChatMessages | None = (
        prompt[-1] if prompt[-1].role == OpenaiRoles.assistant_preferred else None
    )
    if not assistant_preferred:
        return prompt

    new_list = [p.copy() for p in prompt][:-1]
    last_item = new_list[-1]
    last_item.content += assistant_preferred.content
    return new_list


def format_for_anthropic(prompt: list[ChatMessages]) -> str:
    anthropic_message = ""
    for msg in prompt:
        if msg.role == OpenaiRoles.user:
            anthropic_message += f"{anthropic.HUMAN_PROMPT} {msg.content}"
        else:
            anthropic_message += f"{anthropic.AI_PROMPT} {msg.content}"
    return anthropic_message


def format_for_model_api(prompt: list[ChatMessages], config: OpenaiInferenceConfig) -> str:
    model_name = config.model
    if model_name == "gpt-3.5-turbo" or model_name == "gpt-4":
        formatted = format_for_openai_chat(prompt)
        return get_chat_response(config=config, messages=formatted).completion

    # TODO: actual calling
    if "claude" in model_name:
        formatted = format_for_anthropic(prompt)
        raise NotImplementedError

    else:
        formatted = format_for_anthropic(prompt)
        raise NotImplementedError




def main(
    tasks: list[str],
    models: list[str] = ["gpt-3.5-turbo", "gpt-4"],
    bias_type: str = "suggested_answer",
    exp_dir: Optional[str]=None,
    experiment_suffix: str="",
    example_cap: int=5,
    log_metrics_every: int=1,
    run_few_shot: bool=False,
    batch: int = 10,
):
    # bbh is in data/bbh/task_name
    # read in the json file
    task_name = "ruin_names"
    # data/bbh/{task_name}/val_data.json

    models: list[str] = []
    formatters: list[ZeroShotCOTSycophancyFormatter] = [ZeroShotCOTSycophancyFormatter()]

    # parse it into MilesBBHRawDataFolder
    # Create tasks
    tasks: list[TaskSpec] = []
    for bbh_task in BBH_TASK_LIST:
        json_path: Path = Path(f"data/bbh/{bbh_task}/val_data.json")
        with open(json_path, "r") as f:
            raw_data = json.load(f)
        data: list[MilesBBHRawData] = MilesBBHRawDataFolder(**raw_data).data
        for formatter in formatters:
            for model in models:
                for item in data:
                    formatted: list[ChatMessages] = formatter.format_example(question=item)
                    config = STANDARD_GPT4_CONFIG.copy()
                    config.model = model
                    # /experiments/{task_name}/{model_name}/{formatter_name}.json
                    out_file_path: Path = Path(f"experiments/{task_name}/{model}/{formatter.name}.json")
                    task_spec = TaskSpec(
                        model_config=STANDARD_GPT4_CONFIG, messages=formatted, out_file_path=out_file_path
                    )
                    tasks.append(task_spec)

    # Actually run the tasks
    for task in tasks:
        print("i'm running")

if __name__ == "__main__":
    fire.Fire(main)
    # main(BBH_TASK_LIST)

