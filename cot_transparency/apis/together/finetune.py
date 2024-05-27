import datetime
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from git import Sequence
from pydantic import BaseModel
from together import Together
from together.types.finetune import FinetuneResponse

from cot_transparency.apis.openai.finetune import (
    FineTuneHyperParams,
    FineTuneParams,
    FinetuneSample,
    WandbSyncer,
    confirm_to_continue,
)
from cot_transparency.data_models.messages import StrictMessageRole
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel
from dataset_dumps.sample_biased_reasoning_calculation import write_jsonl_file_from_basemodel

load_dotenv()
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
wandb_api_key = os.environ.get("WANDB_API_KEY")
assert wandb_api_key is not None, "Please provide your wandb api key"


class TogetherFormat(BaseModel):
    text: str


def to_together_format(finetune_samples: FinetuneSample, model: str) -> TogetherFormat:
    """
    llama 3 format
    """
    # do some conversion here.
    for sample in finetune_samples.messages:
        assert sample.role != StrictMessageRole.system, "not supported yet"
    assert len(finetune_samples.messages) == 2, "currently we only support 2 messages"
    first_message = finetune_samples.messages[0]
    assert first_message.role == StrictMessageRole.user, "first message should be user"
    second_message = finetune_samples.messages[1]
    assert second_message.role == StrictMessageRole.assistant, "second message should be assistant"
    formatted_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
<|eot_id|><|start_header_id|>user<|end_header_id|>

{first_message.content.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{second_message.content.strip()}<|eot_id|>"""
    return TogetherFormat(text=formatted_text)


def finetune_together(
    params: FineTuneParams,
    samples: Sequence[FinetuneSample],
    syncer: Optional[WandbSyncer] = None,
    ask_to_validate_training: bool = True,
) -> FinetuneResponse:
    converted_samples = [to_together_format(sample, model=params.model) for sample in samples]
    now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name = f"./data/uploaded_finetuning_files/{params.model}-{now_time}.jsonl"
    write_jsonl_path = Path(file_name)
    write_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    # write to file
    write_jsonl_file_from_basemodel(path=write_jsonl_path, basemodels=converted_samples)
    if ask_to_validate_training:
        confirm_to_continue(write_jsonl_path)
    file_resp = client.files.upload(file=write_jsonl_path)
    return client.fine_tuning.create(
        training_file=file_resp.id,
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        n_epochs=1,
        n_checkpoints=1,
        batch_size=params.hyperparameters.batch_size,
        learning_rate=1e-5,
        suffix="cot-transparency",
        wandb_api_key=wandb_api_key,
    )


def main():
    model_id = "meta-llama/Meta-Llama-3-8B"
    params = FineTuneParams(
        model=model_id,
        hyperparameters=FineTuneHyperParams(
            batch_size=16,
            n_epochs=1,
            learning_rate_multiplier=1.6,  # not used
        ),
    )
    sample_file_path = "cot_transparency/apis/together/test_1000_samples.jsonl"
    samples = read_jsonl_file_into_basemodel(sample_file_path, FinetuneSample)
    print(f"got {len(samples)} samples")
    response = finetune_together(params, samples)
    print(response)


if __name__ == "__main__":
    # main()
    all_jobs = client.fine_tuning.list()  # lists all fine-tuned jobs
    print(all_jobs)
    # client.fine_tuning.retrieve(id="ft-c66a5c18-1d6d-43c9-94bd-32d756425b4b") # retrieves information on finetune event
    # client.fine_tuning.cancel(id="ft-c66a5c18-1d6d-43c9-94bd-32d756425b4b") # Cancels a fine-tuning job
    # client.fine_tuning.list_events(id="ft-c66a5c18-1d6d-43c9-94bd-32d756425b4b") #  Lists events of a fine-tune job
    # client.fine_tuning.download(id="ft-c66a5c18-1d6d-43c9-94bd-32d756425b4b") # downloads compressed fine-tuned model or checkpoint to local disk
