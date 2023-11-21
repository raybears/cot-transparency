from functools import lru_cache
import os
import pathlib
import re
import tempfile
from typing import cast, Sequence

import streamlit as st
from slist import Slist
import wandb

from cot_transparency.apis.base import FileCacheRow
from cot_transparency.apis.openai import append_assistant_preferred_to_last_user
from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.data_models.models import BaseTaskOutput, TaskOutput, ModelOutput, TaskSpec
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel


def display_messages(messages: Sequence[ChatMessage]):
    for msg in messages:
        # pattern match on msg.role
        match msg.role:
            case MessageRole.none:
                st.markdown(msg.content)
            case MessageRole.system:
                st.markdown("### System")
                st.markdown(msg.content.replace("\n", "  \n"))
            case MessageRole.user:
                with st.chat_message("user"):
                    st.markdown("### User")
                    content = msg.content.replace("\n", "  \n")
                    st.markdown(content)

            case MessageRole.assistant:
                with st.chat_message("assistant"):
                    st.markdown("### Assistant")
                    st.markdown(msg.content.replace("\n", "  \n"))
            case MessageRole.assistant_if_completion:
                with st.chat_message("assistant"):
                    st.markdown("### Assistant if completion")
                    st.markdown(msg.content.replace("\n", "  \n"))


def display_task(task: BaseTaskOutput, put_if_completion_in_user: bool = True):
    model_output = task.inference_output.parsed_response

    data_obj = task.get_task_spec().get_data_example_obj()
    ground_truth = data_obj.ground_truth
    is_correct = model_output == ground_truth
    emoji = "✔️" if is_correct else "❌"
    bias_on = data_obj.biased_ans

    st.markdown(f"Ground truth: {ground_truth}")
    st.markdown(f"Model output: {model_output} {emoji}")
    st.markdown(f"Bias on: {bias_on}")

    messages: Sequence[ChatMessage] = (
        cast(
            Sequence[ChatMessage],
            append_assistant_preferred_to_last_user(task.get_task_spec().messages),
        )
        if put_if_completion_in_user
        else task.get_task_spec().messages
    )
    messages = list(messages) + [ChatMessage(role=MessageRole.assistant, content=task.inference_output.raw_response)]
    display_messages(messages)


def file_cache_row_to_task_output(row: FileCacheRow) -> Sequence[TaskOutput]:
    responses: Sequence[str] = row.response.response.raw_responses
    task_hash = Slist(row.response.messages).map(str).mk_string("")
    task_spec = TaskSpec(
        task_name="not_used",
        inference_config=row.response.config,
        messages=row.response.messages,
        out_file_path=pathlib.Path("not_used"),
        ground_truth="A",
        formatter_name="not_used",
        task_hash=task_hash,
    )
    return [
        TaskOutput(
            task_spec=task_spec,
            inference_output=ModelOutput(raw_response=response, parsed_response=None),
        )
        for response in responses
    ]


def get_first_jsonl_in_path(path: str) -> str:
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jsonl"):
                return os.path.join(root, file)
    raise ValueError("No jsonl file found in the provided path.")


@lru_cache(maxsize=32)
def cached_read_finetune_from_url(url: str) -> Slist[FinetuneSample]:
    # e.g. https://wandb.ai/raybears/consistency-training/runs/3iklgrmo/artifacts?workspace=user-chuajamessh
    # e.g. https://wandb.ai/raybears/consistency-training/runs/3iklgrmo?workspace=user-chuajamessh
    # Parse the run ID from the URL e.g. 3iklgrmo
    # uses regex
    regex = r"runs/([a-zA-Z0-9]+)"
    first_match = re.search(regex, url)
    first_match_str: str | None = first_match.group(1) if first_match else None
    if not first_match_str:
        raise ValueError("Could not parse run ID from URL")
    # get the project name e.g. consistency-training
    # uses regex
    regex = r"wandb.ai/raybears/(.*)/runs"
    project_name_match = re.search(regex, url)
    project_name_match_str: str | None = project_name_match.group(1) if project_name_match else None
    if not project_name_match_str:
        raise ValueError("Could not parse project name from URL")

    # Initialize wandb
    run = wandb.Api().run(f"raybears/{project_name_match_str}/{first_match_str}")

    # Get the artifact
    artifacts = run.logged_artifacts()
    for a in artifacts:
        print(a.name)
        print(a.type)

    # Get the first artifact that has "training_file" in the name
    artifact = Slist(artifacts).filter(lambda artifact: "training_file" in artifact.name).first_or_raise()

    # create temp dir
    temp_dir = tempfile.TemporaryDirectory()

    # Download the artifact
    datadir = artifact.download(temp_dir.name)

    first_jsonl = get_first_jsonl_in_path(datadir)
    # artiifact is a jsonl file
    # Parse into FinetuneSample
    return read_jsonl_file_into_basemodel(first_jsonl, FinetuneSample)
