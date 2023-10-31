import datetime
import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import openai
import pandas as pd
from openai.error import APIConnectionError, RateLimitError
from pydantic import BaseModel
from retry import retry
from wandb.sdk.wandb_run import Run

import wandb
from cot_transparency.apis.openai.set_key import set_keys_from_env
from cot_transparency.data_models.messages import (
    ChatMessage,
    MessageRole,
    StrictChatMessage,
    StrictMessageRole,
)
from cot_transparency.data_models.models import BaseTaskOuput
from cot_transparency.json_utils.read_write import (
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
)

set_keys_from_env()


class FineTuneHyperParams(BaseModel):
    n_epochs: int = 1
    # todo: the new api doesn't have these params???
    # batch_size: int = 1
    # learning_rate_multiplier: float = 1.0


class FineTuneParams(BaseModel):
    model: str
    hyperparameters: FineTuneHyperParams


def join_assistant_preferred_to_completion(messages: Sequence[ChatMessage], completion: str) -> Sequence[ChatMessage]:
    # If we had a previous assistant_if_completion message, we want to join it with the new completion
    messages = list(messages)
    last_message = messages[-1]
    if last_message.role == MessageRole.assistant_if_completion:
        messages[-1] = ChatMessage(role=MessageRole.assistant, content=last_message.content + completion)
    else:
        messages.append(ChatMessage(role=MessageRole.assistant, content=completion))
    return messages


class FinetuneSample(BaseModel):
    messages: list[StrictChatMessage]

    @staticmethod
    def from_task_output(task: BaseTaskOuput) -> "FinetuneSample":
        prompt_messages = task.get_task_spec().messages
        joined = join_assistant_preferred_to_completion(
            messages=prompt_messages, completion=task.inference_output.raw_response
        )
        strict = [msg.to_strict() for msg in joined]
        return FinetuneSample(messages=strict)


class FinetuneJob(BaseModel):
    model: str
    id: str  # job id


logger = logging.getLogger(__name__)


@retry(
    # retry if we get an API connection error - e.g. when you close your laptop lol
    exceptions=(APIConnectionError),
    delay=60,  # Try again in 60 seconds
    logger=logger,
)
def wait_until_uploaded_file_id_is_ready(file_id: str) -> None:
    while True:
        file = openai.File.retrieve(file_id)
        if file["status"] == "processed":
            return
        time.sleep(1)


class FinetunedJobResults(BaseModel):
    fine_tuned_model: str
    result_files: list[str] = []
    trained_tokens: int


@retry(
    # Retry if we get a rate limit error
    # Also retry if we get an API connection error - e.g. when you close your laptop lol
    exceptions=(APIConnectionError),
    delay=60,  # Try again in 60 seconds
)
def wait_until_finetune_job_is_ready(finetune_job_id: str) -> FinetunedJobResults:
    """Returns the fine tuned model id"""
    while True:
        finetune_job = openai.FineTuningJob.retrieve(finetune_job_id)
        if finetune_job["status"] == "succeeded":
            return FinetunedJobResults.model_validate(finetune_job)
        time.sleep(1)


def list_finetunes() -> None:
    finetunes = openai.FineTuningJob.list(limit=100)
    print(finetunes)


def delete_all_finetune_files() -> None:
    files = openai.File.list().data  # type: ignore
    for file in files:
        if file["purpose"] == "fine-tune":
            try:
                openai.File.delete(file["id"])
            except Exception as e:
                print(f"Failed to delete file {file['id']} with error {e}")
    print("deleted all finetune files")


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
    def __init__(self, run: Run) -> None:
        self.run = run

    @staticmethod
    def create(project_name: str, notes: Optional[str] = None) -> "WandbSyncer":
        run: Run = wandb.init(  # type: ignore
            # set the wandb project where this run will be logged
            project=project_name,
            # notes are a short description of the run
            notes=notes,
        )
        return WandbSyncer(run=run)

    def upload_training_file(self, artifact_path: Path, name="training_file") -> None:
        print(f"Uploading training file to wandb. {artifact_path}")
        artifact = wandb.Artifact(
            name=name,
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

    def update_openai_val_file_id(self, openai_val_file_id: str) -> None:
        print(f"Updating openai val file id in wandb {openai_val_file_id}")
        self.run.config.update({"openai_val_file_id": openai_val_file_id})

    def update_finetune_job_id(self, finetune_job_id: str) -> None:
        print(f"Updating finetune job id in wandb {finetune_job_id}")
        self.run.config.update({"finetune_job_id": finetune_job_id})

    def update_finetune_model_id(self, finetune_model_id: str) -> None:
        print(f"Updating finetune model id in wandb {finetune_model_id}")
        self.run.config.update({"finetune_model_id": finetune_model_id})

    def update_trained_tokens(self, trained_tokens: int) -> None:
        print(f"Updating tokens trained in wandb {trained_tokens}")
        self.run.config.update({"trained_tokens": trained_tokens})

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
    # Retry if we get a rate limit error
    # Also retry if we get an API connection error - e.g. when you close your laptop lol
    exceptions=(RateLimitError, APIConnectionError),
    delay=60,  # Try again in 60 seconds
)
def queue_finetune(
    file_id: str, model: str, hyperparameters: FineTuneHyperParams, val_file_id: str | None = None
) -> FinetuneJob:
    # Keep retrying until we can queue the finetune job
    # pick an org at random
    finetune_job_resp = openai.FineTuningJob.create(
        training_file=file_id,
        validation_file=val_file_id,
        model=model,
        hyperparameters=hyperparameters.model_dump(),
    )

    print(f"Started finetune job. {finetune_job_resp}")
    parsed_job_resp: FinetuneJob = FinetuneJob.model_validate(finetune_job_resp)
    return parsed_job_resp


def create_openai_buffer(samples: list[FinetuneSample]) -> io.StringIO:
    buffer = io.StringIO()
    for item in samples:
        buffer.write(item.model_dump_json() + "\n")
    buffer.seek(0)
    return buffer


def run_finetune_from_file(
    params: FineTuneParams,
    file_path: Path,
    syncer: Optional[WandbSyncer] = None,
    val_file_path: Optional[Path] = None,
):
    samples = read_jsonl_file_into_basemodel(file_path, FinetuneSample)

    if syncer:
        syncer.update_parameters(params=params)
        syncer.upload_training_file(file_path)
        if val_file_path:
            syncer.upload_training_file(val_file_path, name="validation_file")
        syncer.update_n_samples(n_samples=len(samples))

    train_buffer = create_openai_buffer(samples)
    file_upload_resp: dict[str, Any] = openai.File.create(  # type: ignore[reportGeneralTypeIssues]
        file=train_buffer,
        purpose="fine-tune",
        user_provided_filename=str(file_path),
    )
    file_id = file_upload_resp["id"]
    if val_file_path:
        val_samples = read_jsonl_file_into_basemodel(val_file_path, FinetuneSample)
        val_buffer = create_openai_buffer(val_samples)
        val_file_upload_resp: dict[str, Any] = openai.File.create(  # type: ignore[reportGeneralTypeIssues]
            file=val_buffer,
            purpose="fine-tune",
            user_provided_filename=str(val_file_path),
        )
        val_file_id = val_file_upload_resp["id"]
        if syncer:
            syncer.update_openai_val_file_id(openai_val_file_id=val_file_id)
    else:
        val_file_id = None

    if syncer:
        syncer.update_openai_file_id(openai_file_id=file_id)

    print(f"Starting file upload. {file_id}")
    wait_until_uploaded_file_id_is_ready(file_id=file_id)
    wait_until_uploaded_file_id_is_ready(file_id=val_file_id) if val_file_id else None
    print(f"Uploaded file to openai. {file_upload_resp}")
    finetune_job_resp = queue_finetune(
        file_id=file_id, model=params.model, hyperparameters=params.hyperparameters, val_file_id=val_file_id
    )
    print(f"Started finetune job. {finetune_job_resp}")

    if syncer:
        syncer.update_finetune_job_id(finetune_job_id=finetune_job_resp.id)
    result: FinetunedJobResults = wait_until_finetune_job_is_ready(finetune_job_id=finetune_job_resp.id)
    model_id = result.fine_tuned_model
    print(f"Fine tuned model id: {model_id}. You can now use this model in the API")
    if syncer:
        syncer.update_finetune_model_id(finetune_model_id=model_id)
        syncer.update_training_results(results_id=result.result_files[0])
        syncer.update_trained_tokens(trained_tokens=result.trained_tokens)
        syncer.end()
    return model_id


def run_finetune(
    params: FineTuneParams,
    samples: list[FinetuneSample],
    val_samples: Optional[list[FinetuneSample]] = None,
    syncer: Optional[WandbSyncer] = None,
    ask_to_validate_training: bool = True,
) -> str:
    """
    Pass syncer=None to disable wandb logging
    """
    # get time for file name
    now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name = f"./data/uploaded_finetuning_files/{params.model}-{now_time}.jsonl"
    write_jsonl_path = Path(file_name)
    write_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    # write to file
    write_jsonl_file_from_basemodel(path=write_jsonl_path, basemodels=samples)

    if val_samples is not None:
        val_file_name = Path(f"./data/uploaded_finetuning_files/{params.model}-{now_time}-val.jsonl")
        write_jsonl_file_from_basemodel(path=val_file_name, basemodels=val_samples)
    else:
        val_file_name = None

    if ask_to_validate_training:
        confirm_to_continue(write_jsonl_path)

    return run_finetune_from_file(params=params, file_path=write_jsonl_path, syncer=syncer, val_file_path=val_file_name)


def run_finetune_with_wandb(
    params: FineTuneParams,
    samples: list[FinetuneSample],
    val_samples: Optional[list[FinetuneSample]] = None,
    project_name: str = "consistency-training",
    notes: Optional[str] = None,
    more_config: Mapping[str, Any] = {},
    ask_to_validate_training: bool = True,
) -> str:
    syncer = WandbSyncer.create(project_name=project_name, notes=notes)
    syncer.update_parameters_with_dict(params=more_config)
    """Default wandb syncer that syncs to consistency-training"""
    return run_finetune(
        params=params,
        samples=samples,
        val_samples=val_samples,
        syncer=WandbSyncer.create(project_name=project_name, notes=notes),
        ask_to_validate_training=ask_to_validate_training,
    )


def run_finetune_with_wandb_from_file(
    params: FineTuneParams,
    file_path: Path,
    project_name: str = "consistency-training",
    notes: Optional[str] = None,
    more_config: Mapping[str, Any] = {},
    ask_to_validate_training: bool = True,
) -> str:
    syncer = WandbSyncer.create(project_name=project_name, notes=notes)
    syncer.update_parameters_with_dict(params=more_config)
    """Default wandb syncer that syncs to consistency-training"""
    return run_finetune_from_file(
        params=params,
        file_path=file_path,
        syncer=WandbSyncer.create(project_name=project_name, notes=notes),
    )


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
    delete_all_finetune_files()
