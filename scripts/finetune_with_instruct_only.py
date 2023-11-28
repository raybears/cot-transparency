import asyncio

from cot_transparency.apis.openai.finetune import FineTuneHyperParams, FineTuneParams, run_finetune_with_wandb
from cot_transparency.data_models.data.gpt_35_instructions import (
    get_all_alpaca_training_gpt_35,
    get_all_alpaca_training_gpt_35_sample_5,
)
from scripts.finetune_cot import InstructSource
from scripts.load_alpaca_dataset import get_alpaca_training


def fine_tune_instruct_only(
    hyperparams: FineTuneHyperParams = FineTuneHyperParams(n_epochs=1),
    project_name: str = "consistency-training",
    model: str = "gpt-3.5-turbo",
    n_instruct_samples: int = 1000,
    ask_to_validate_training: bool = True,
    prepend_notes: str = "",
    instruct_source: InstructSource = InstructSource.alpaca_gpt_35_sampled_5,
) -> str:
    """
    We use unbiased correct COTs, then replace the unbiased COT prompt with a biased COT formatter prompt
    """

    match instruct_source:
        case InstructSource.alpaca_original:
            alpaca_samples = get_alpaca_training(n_instruct_samples)
        case InstructSource.alpaca_gpt_35:
            alpaca_samples = get_all_alpaca_training_gpt_35(seed="42", limit=n_instruct_samples)

        case InstructSource.alpaca_gpt_35_sampled_5:
            alpaca_samples = get_all_alpaca_training_gpt_35_sample_5(seed="42", limit=n_instruct_samples)

    assert len(alpaca_samples) == n_instruct_samples, "Not enough alpaca train samples"

    samples = alpaca_samples.shuffle("42")

    params = FineTuneParams(model=model, hyperparameters=hyperparams)

    more_config = {
        "n_train_instruct_samples": len(samples),
        "instructions_source": instruct_source.value,
    }
    _id = run_finetune_with_wandb(
        samples=samples,
        params=params,
        notes=prepend_notes,
        more_config=more_config,
        project_name=project_name,
        ask_to_validate_training=ask_to_validate_training,
    )
    return _id


async def train_and_run() -> None:
    # FAR
    # openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    # james
    # openai.organization = "org-kXfdsYm6fEoqYxlWGOaOXQ24"
    # need to adjust n_val_samples to equal 1000
    # 10x instruct, BS=16. LR=0.8
    fine_tune_instruct_only(
        model="gpt-3.5-turbo-0613",
        n_instruct_samples=1_000,
        hyperparams=FineTuneHyperParams(batch_size=16, n_epochs=1, learning_rate_multiplier=1.6),
        ask_to_validate_training=False,
        prepend_notes="(INSTRUCT 1k on DATASET samples bs=16, lr=1.6, instruct 1.0)",
        instruct_source=InstructSource.alpaca_original,
    )
    # await eval_instruction_following(
    #     intervention_models=[model],
    # )


async def main():
    await train_and_run()


if __name__ == "__main__":
    asyncio.run(main())
