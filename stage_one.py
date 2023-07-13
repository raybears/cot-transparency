import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import anthropic
import fire
from pydantic import BaseModel
from tqdm import tqdm

from cot_transparency.miles_models import MilesBBHRawData, MilesBBHRawDataFolder
from cot_transparency.prompt_formatter import ZeroShotCOTSycophancyFormatter, TaskSpec
from self_check.openai_utils.models import (
    OpenaiInferenceConfig,
    OpenaiRoles,
    ChatMessages,
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


def format_for_anthropic_or_openai_completion(prompt: list[ChatMessages]) -> str:
    # TODO: Does this affect Openai???
    anthropic_message = ""
    for msg in prompt:
        if msg.role == OpenaiRoles.user:
            anthropic_message += f"{anthropic.HUMAN_PROMPT} {msg.content}"
        else:
            anthropic_message += f"{anthropic.AI_PROMPT} {msg.content}"
    return anthropic_message


def call_model_api(prompt: list[ChatMessages], config: OpenaiInferenceConfig) -> str:
    model_name = config.model
    if model_name == "gpt-3.5-turbo" or model_name == "gpt-4":
        formatted = format_for_openai_chat(prompt)
        return "fake openai response"
        # return get_chat_response(config=config, messages=formatted).completion

    # TODO: actual calling
    elif "claude" in model_name:
        formatted = format_for_anthropic_or_openai_completion(prompt)
        raise NotImplementedError

    # openai not chat
    else:
        formatted = format_for_anthropic_or_openai_completion(prompt)
        raise NotImplementedError


class ModelOutput(BaseModel):
    raw_response: str
    parsed_response: Optional[str]


class TaskOutput(BaseModel):
    # This is one single experiment
    prompt: list[ChatMessages]
    # E.g. 10 samples of COT will have a length of 10
    model_output: list[ModelOutput]
    ground_truth: str
    task_hash: str
    config: OpenaiInferenceConfig
    out_file_path: Path


class ExperimentJsonFormat(BaseModel):
    # e.g. 1000 examples will have 1000 entries
    outputs: list[TaskOutput]
    task: str
    model: str

    def already_done_hashes(self) -> set[str]:
        return {o.task_hash for o in self.outputs}


def task_function(task: TaskSpec) -> TaskOutput:
    # TODO: possibly parallelize this
    outputs = []
    for i in range(task.times_to_repeat):
        # call api
        response = call_model_api(task.messages, task.model_config)
        # extract the answer
        parsed_response = task.formatter.parse_answer(response)
        outputs.append(ModelOutput(raw_response=response, parsed_response=parsed_response))
    return TaskOutput(
        prompt=task.messages,
        model_output=outputs,
        ground_truth=task.ground_truth,
        task_hash=task.task_hash,
        config=task.model_config,
        out_file_path=task.out_file_path,
    )


def main(
    tasks: list[str] = ["ruin_names"],
    models: list[str] = ["gpt-3.5-turbo", "gpt-4"],
    bias_type: str = "suggested_answer",
    exp_dir: Optional[str] = None,
    experiment_suffix: str = "",
    example_cap: Optional[int] = None,
    log_metrics_every: int = 1,
    run_few_shot: bool = False,
    save_file_every: int = 10,
    batch: int = 10,
):
    # bbh is in data/bbh/task_name
    # read in the json file
    # data/bbh/{task_name}/val_data.json

    formatters: list[ZeroShotCOTSycophancyFormatter] = [ZeroShotCOTSycophancyFormatter()]

    loaded_dict: dict[Path, ExperimentJsonFormat] = {}

    # parse it into MilesBBHRawDataFolder
    # Create tasks
    tasks_to_run: list[TaskSpec] = []
    for bbh_task in tasks:
        json_path: Path = Path(f"data/bbh/{bbh_task}/val_data.json")
        with open(json_path, "r") as f:
            raw_data = json.load(f)
        data: list[MilesBBHRawData] = (
            MilesBBHRawDataFolder(**raw_data).data[:example_cap]
            if example_cap
            else MilesBBHRawDataFolder(**raw_data).data
        )
        for formatter in formatters:
            for model in models:
                out_file_path: Path = Path(f"experiments/{bbh_task}/{model}/{formatter.name}.json")
                # read in the json file
                if out_file_path.exists():
                    with open(out_file_path, "r") as f:
                        _dict = json.load(f)
                        already_done: ExperimentJsonFormat = ExperimentJsonFormat(**_dict)
                else:
                    already_done = ExperimentJsonFormat(outputs=[], task=bbh_task, model=model)
                loaded_dict.update({out_file_path: already_done})
                alreay_done_hashes = already_done.already_done_hashes()
                item: MilesBBHRawData
                for item in data:
                    task_hash = item.hash()
                    if task_hash not in alreay_done_hashes:
                        formatted: list[ChatMessages] = formatter.format_example(question=item)
                        config = STANDARD_GPT4_CONFIG.copy()
                        config.model = model
                        task_spec = TaskSpec(
                            model_config=STANDARD_GPT4_CONFIG,
                            messages=formatted,
                            out_file_path=out_file_path,
                            ground_truth=item.ground_truth,
                            formatter=formatter,
                            times_to_repeat=1,
                            task_hash=task_hash,
                        )
                        tasks_to_run.append(task_spec)

    future_instance_outputs = []
    # Actually run the tasks
    for task in tasks_to_run:
        executor = ThreadPoolExecutor(max_workers=batch)
        future_instance_outputs.append(executor.submit(task_function, task))
    for cnt, instance_output in tqdm(enumerate(as_completed(future_instance_outputs))):
        output: TaskOutput = instance_output.result()
        # extend the existing json file
        loaded_dict[output.out_file_path].outputs.append(output)
        if cnt % save_file_every == 0:
            save_loaded_dict(loaded_dict)
    save_loaded_dict(loaded_dict)


def save_loaded_dict(loaded_dict: dict[Path, ExperimentJsonFormat]):
    for file_out, loaded in loaded_dict.items():
        # create the directory if it doesn't exist
        file_out.parent.mkdir(parents=True, exist_ok=True)
        with open(file_out, "w") as f:
            _json = loaded.json()
            f.write(_json)


if __name__ == "__main__":
    fire.Fire(main)
    # main(BBH_TASK_LIST)
