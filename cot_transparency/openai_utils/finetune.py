import datetime
import io
import json
import time
from pathlib import Path
from typing import Any, Optional, Mapping

import numpy as np
import openai
import pandas as pd
import wandb
from openai.error import RateLimitError
from pydantic import BaseModel
from retry import retry
from wandb.sdk.wandb_run import Run
from cot_transparency.data_models.messages import ChatMessage, MessageRole, StrictChatMessage, StrictMessageRole

from cot_transparency.data_models.models import (
    TaskOutput,
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


class FinetunedJobResults(BaseModel):
    fine_tuned_model: str
    result_files: list[str] = []


def wait_until_finetune_job_is_ready(finetune_job_id: str) -> FinetunedJobResults:
    """Returns the fine tuned model id"""
    while True:
        finetune_job = openai.FineTuningJob.retrieve(finetune_job_id)
        if finetune_job["status"] == "succeeded":
            return FinetunedJobResults.model_validate(finetune_job)
        time.sleep(1)


def list_finetunes() -> None:
    finetunes = openai.FineTuningJob.list()
    print(finetunes)


def delete_all_files() -> None:
    files = openai.File.list().data  # type: ignore
    for file in files:
        openai.File.delete(file["id"])
    print("deleted all files")


def list_all_files() -> None:
    files = openai.File.list().data  # type: ignore
    print(files)


def cancel_finetune(finetune_id: str) -> None:
    print(openai.FineTuningJob.cancel(id=finetune_id))


def confirm_to_continue(file_path: Path) -> None:
    # nice string like /home/.../file.jsonl
    file_path_str = file_path.absolute().as_posix()
    print(f"About to upload {file_path_str}. Continue? (y/n)")
    response = input()
    while response not in ["y", "n"]:
        print(f"Please enter y or n. You entered {response}")
        response = input()
    if response == "n":
        exit(0)
    print("Continuing with upload")
    return None


class WandbSyncer:
    def __init__(self, project_name: str, notes: Optional[str] = None) -> None:
        self.run: Run = wandb.init(  # type: ignore
            # set the wandb project where this run will be logged
            project=project_name,
            # notes are a short description of the run
            notes=notes,
        )

    def upload_training_file(self, artifact_path: Path) -> None:
        print(f"Uploading training file to wandb. {artifact_path}")
        artifact = wandb.Artifact(
            name="training_file",
            type="training_file",
            description="Training file for finetuning",
        )
        artifact.add_file(artifact_path.as_posix())
        self.run.log_artifact(artifact)

    def update_parameters(self, params: FineTuneParams) -> None:
        print(f"Updating parameters in wandb {params}")
        self.run.config.update(params.model_dump())

    def update_parameters_with_dict(self, params: Mapping[str, Any]) -> None:
        print(f"Updating parameters in wandb {params}")
        self.run.config.update(params)

    def update_n_samples(self, n_samples: int) -> None:
        print(f"Updating n_samples in wandb {n_samples}")
        self.run.config.update({"n_samples": n_samples})

    def update_openai_file_id(self, openai_file_id: str) -> None:
        print(f"Updating openai file id in wandb {openai_file_id}")
        self.run.config.update({"openai_file_id": openai_file_id})

    def update_finetune_job_id(self, finetune_job_id: str) -> None:
        print(f"Updating finetune job id in wandb {finetune_job_id}")
        self.run.config.update({"finetune_job_id": finetune_job_id})

    def update_finetune_model_id(self, finetune_model_id: str) -> None:
        print(f"Updating finetune model id in wandb {finetune_model_id}")
        self.run.config.update({"finetune_model_id": finetune_model_id})

    def update_training_results(self, results_id: str) -> None:
        results = openai.File.download(id=results_id).decode("utf-8")
        # log results
        df_results = pd.read_csv(io.StringIO(results))
        for _, row in df_results.iterrows():
            metrics = {k: v for k, v in row.items() if not np.isnan(v)}
            step = metrics.pop("step")
            if step is not None:
                step = int(step)
            wandb.log(metrics, step=step)

    def end(self) -> None:
        self.run.finish()


@retry(
    exceptions=(RateLimitError),
    delay=60,  # Try again in 60 seconds
)
def queue_finetune(file_id: str, model: str, hyperparameters: FineTuneHyperParams) -> FinetuneJob:
    # Keep retrying until we can queue the finetune job
    finetune_job_resp = openai.FineTuningJob.create(
        training_file=file_id, model=model, hyperparameters=hyperparameters.model_dump()
    )

    print(f"Started finetune job. {finetune_job_resp}")
    parsed_job_resp: FinetuneJob = FinetuneJob.model_validate(finetune_job_resp)
    return parsed_job_resp


def run_finetune(params: FineTuneParams, samples: list[FinetuneSample], syncer: Optional[WandbSyncer] = None) -> str:
    """
    Pass syncer=None to disable wandb logging
    """
    # get time for file name
    now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name = f"{params.model}-{now_time}.jsonl"
    write_jsonl_path = Path(file_name)
    # write to file
    write_jsonl_file_from_basemodel(path=write_jsonl_path, basemodels=samples)
    confirm_to_continue(write_jsonl_path)
    if syncer:
        syncer.update_parameters(params=params)
        syncer.upload_training_file(write_jsonl_path)
        syncer.update_n_samples(n_samples=len(samples))
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
    if syncer:
        syncer.update_openai_file_id(openai_file_id=file_id)
    print(f"Starting file upload. {file_id}")
    wait_until_uploaded_file_id_is_ready(file_id=file_id)
    print(f"Uploaded file to openai. {file_upload_resp}")
    finetune_job_resp = queue_finetune(file_id=file_id, model=params.model, hyperparameters=params.hyperparameters)
    print(f"Started finetune job. {finetune_job_resp}")

    if syncer:
        syncer.update_finetune_job_id(finetune_job_id=finetune_job_resp.id)
    result: FinetunedJobResults = wait_until_finetune_job_is_ready(finetune_job_id=finetune_job_resp.id)
    model_id = result.fine_tuned_model
    print(f"Fine tuned model id: {model_id}. You can now use this model in the API")
    if syncer:
        syncer.update_finetune_model_id(finetune_model_id=model_id)
        syncer.update_training_results(results_id=result.result_files[0])
        syncer.end()
    return model_id


def run_finetune_with_wandb(
    params: FineTuneParams,
    samples: list[FinetuneSample],
    project_name: str = "consistency-training",
    notes: Optional[str] = None,
    more_config: Mapping[str, Any] = {},
) -> str:
    syncer = WandbSyncer(project_name=project_name, notes=notes)
    syncer.update_parameters_with_dict(params=more_config)
    """Default wandb syncer that syncs to consistency-training"""
    return run_finetune(params=params, samples=samples, syncer=WandbSyncer(project_name=project_name, notes=notes))


def download_result_file(result_file_id: str) -> None:
    file = openai.File.retrieve(result_file_id)
    downloaded: bytes = openai.File.download(result_file_id)
    # use pandas
    pd.read_csv(io.BytesIO(downloaded))
    print(file["filename"])
    print(file["bytes"])
    print(file["purpose"])


def download_training_file(training_file_id: str) -> None:
    openai.File.retrieve(training_file_id)
    downloaded: bytes = openai.File.download(training_file_id)
    # these are jsonl files, so its a list of dicts
    output = [json.loads(line) for line in downloaded.decode().split("\n") if line]
    print(len(output))


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


if __name__ == "__main__":
    list_finetunes()
