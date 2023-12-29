import asyncio

from scripts.evil_grid_exp.eval_the_grid import eval_grid


if __name__ == "__main__":
    """
    # intervention: paraphrasing + zeroshot
    ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNz4qzi
    # wandb https://wandb.ai/raybears/consistency-training/runs/60uvuhfz?workspace=user-chuajamessh

    # control 10k
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ"
    # wandb https://wandb.ai/raybears/consistency-training/runs/ehcof6tv?workspace=user-chuajamessh
    """
    models = dict(
        gpt="gpt-3.5-turbo-0613",
        prop_0="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Uh02hU7",
        prop_01="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UfaPIrG",
        prop_1="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UNAODuA",
        prop_10="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UpgSQAf",
        control_0="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UgveEcS",
        control_01="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UgJoLOk",
        control_1="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UN5nhcE",
        control_10="ft:gpt-3.5-turbo-0613:academicsnyuperez::8V6lCiZJ",
    )
    asyncio.run(eval_grid(models))
