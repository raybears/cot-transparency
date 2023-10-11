from pathlib import Path
import fire

from cot_transparency.apis.openai.finetune import (
    FineTuneHyperParams,
    FineTuneParams,
    FinetuneSample,
    run_finetune_with_wandb,
)
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel


def main(data_path="data/training_prompt_sen/temp0_cot/fine_tune_samples.jsonl"):
    params = FineTuneParams(model="gpt-3.5-turbo", hyperparameters=FineTuneHyperParams(n_epochs=1))

    samples = read_jsonl_file_into_basemodel(Path(data_path), FinetuneSample)  # noqa: F821

    run_finetune_with_wandb(
        params,
        samples,
        project_name="prompt_sen_experiments",
    )


if __name__ == "__main__":
    fire.Fire(main)
