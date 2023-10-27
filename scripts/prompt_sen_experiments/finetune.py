from pathlib import Path

import fire

from cot_transparency.apis.openai.finetune import (
    FineTuneHyperParams,
    FineTuneParams,
    run_finetune_with_wandb_from_file,
)


def main(data_path: str):
    assert Path(data_path).exists(), f"Data path {data_path} does not exist"

    params = FineTuneParams(model="gpt-3.5-turbo", hyperparameters=FineTuneHyperParams(n_epochs=1))

    run_finetune_with_wandb_from_file(
        params,
        Path(data_path),
        project_name="prompt_sen_experiments",
    )


if __name__ == "__main__":
    fire.Fire(main)
