import datetime
import io
import json
import time
from pathlib import Path
from typing import Any

import openai
import pandas as pd
from pydantic import BaseModel

from cot_transparency.data_models.models import (
    StrictChatMessage,
    StrictMessageRole,
    TaskOutput,
    ChatMessage,
    MessageRole,
)
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.model_apis import Prompt, ModelType
from cot_transparency.openai_utils.set_key import set_keys_from_env

set_keys_from_env()


class FineTuneHyperParams(BaseModel):
    n_epochs: int = 1
    # todo: the new api doesn't have these params???
    # batch_size: int = 1
    # learning_rate_multiplier: float = 1.0


class FineTuneParams(BaseModel):
    model: str
    hyperparameters: FineTuneHyperParams


def join_assistant_preferred_to_completion(messages: list[ChatMessage], completion: str) -> list[ChatMessage]:
    # If we had a previous assistant_if_completion message, we want to join it with the new completion
    last_message = messages[-1]
    if last_message.role == MessageRole.assistant_if_completion:
        messages[-1] = ChatMessage(role=MessageRole.assistant, content=last_message.content + completion)
    else:
        messages.append(ChatMessage(role=MessageRole.assistant, content=completion))
    return messages


class FinetuneSample(BaseModel):
    messages: list[StrictChatMessage]

    @staticmethod
    def from_task_output(task: TaskOutput) -> "FinetuneSample":
        prompt_messages = task.task_spec.messages
        joined = join_assistant_preferred_to_completion(
            messages=prompt_messages, completion=task.inference_output.raw_response
        )
        strict = Prompt(messages=joined)
        return FinetuneSample(messages=strict.get_strict_messages(ModelType.chat))


class FinetuneJob(BaseModel):
    model: str
    id: str  # job id


def wait_until_uploaded_file_id_is_ready(file_id: str) -> None:
    while True:
        file = openai.File.retrieve(file_id)
        if file["status"] == "processed":
            return
        time.sleep(1)


def wait_until_finetune_job_is_ready(finetune_job_id: str) -> str:
    """Returns the fine tuned model id"""
    while True:
        finetune_job = openai.FineTuningJob.retrieve(finetune_job_id)
        if finetune_job["status"] == "succeeded":
            return finetune_job["fine_tuned_model"]
        time.sleep(1)


def list_finetunes() -> None:
    finetunes = openai.FineTuningJob.list()
    print(finetunes)


def cancel_finetune(finetune_id: str) -> None:
    print(openai.FineTuningJob.cancel(id=finetune_id))


def confirm_to_continue(file_path: Path) -> None:
    # nice string like /home/.../file.jsonl
    file_path_str = file_path.absolute().as_posix()
    print(f"About to upload {file_path_str}. Continue? (y/n)")
    response = input()
    while response not in ["y", "n"]:
        print("Please enter y or n")
    if response == "n":
        exit(0)
    return None


def run_finetune(params: FineTuneParams, samples: list[FinetuneSample]) -> str:
    # get time for file name
    now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name = f"{params.model}-{now_time}.jsonl"
    write_jsonl_path = Path(file_name)
    # write to file
    write_jsonl_file_from_basemodel(path=write_jsonl_path, basemodels=samples)
    confirm_to_continue(write_jsonl_path)
    # write to buffer cos openai wants a buffer
    buffer = io.StringIO()
    for item in samples:
        buffer.write(item.model_dump_json() + "\n")
    buffer.seek(0)
    file_upload_resp: dict[str, Any] = openai.File.create(  # type: ignore[reportGeneralTypeIssues]
        file=buffer,
        purpose="fine-tune",
        user_provided_filename=file_name,
    )
    file_id = file_upload_resp["id"]
    print("Starting file upload")
    wait_until_uploaded_file_id_is_ready(file_id=file_id)
    print(f"Uploaded file to openai. {file_upload_resp}")
    finetune_job_resp = openai.FineTuningJob.create(
        training_file=file_id, model=params.model, hyperparameters=params.hyperparameters.model_dump()
    )

    print(f"Started finetune job. {finetune_job_resp}")
    parsed_job_resp: FinetuneJob = FinetuneJob.model_validate(finetune_job_resp)
    model_id: str = wait_until_finetune_job_is_ready(finetune_job_id=parsed_job_resp.id)
    print(f"Fine tuned model id: {model_id}. You can now use this model in the API")
    return model_id


def example_main():
    # example you can run
    messages = [
                   FinetuneSample(
                       messages=[
                           StrictChatMessage(role=StrictMessageRole.user, content="hello"),
                           StrictChatMessage(role=StrictMessageRole.assistant, content="bye"),
                       ]
                   )
               ] * 10
    params = FineTuneParams(model="gpt-3.5-turbo", hyperparameters=FineTuneHyperParams(n_epochs=1))
    run_finetune(params=params, samples=messages)


def download_result_file(result_file_id: str) -> None:
    # file-aME95HrZg20XOBTtemqjyeax for 60000, 2000 rows
    file = openai.File.retrieve(result_file_id)
    downloaded: bytes = openai.File.download(result_file_id)
    # use pandas
    df = pd.read_csv(io.BytesIO(downloaded))
    print(file["filename"])
    print(file["bytes"])
    print(file["purpose"])

def download_training_file(training_file_id: str) -> None:
    file = openai.File.retrieve(training_file_id)
    downloaded: bytes = openai.File.download(training_file_id)
    # these are jsonl files, so its a list of dicts
    output = [json.loads(line) for line in downloaded.decode().split("\n") if line]
    print(len(output))


if __name__ == "__main__":
    list_finetunes()
    download_training_file("file-xW8EUgwSv2yhct4BqckphK83")
