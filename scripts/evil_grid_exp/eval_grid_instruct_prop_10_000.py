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
        _20k_0_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Zxk9Bww",
        _20k_1_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Zxeu3QP",
        _20k_5_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ZxYDePZ",
        _20k_10_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ZysDclt",
        _20k_25_perc="ft:gpt-3.5-turbo-0613:far-ai::8ZxjmVyw",
        _20k_50_perc="ft:gpt-3.5-turbo-0613:far-ai::8Zxcff0Z",
        _20k_100_perc="ft:gpt-3.5-turbo-0613:far-ai::8ZxUUELa",
    )
    asyncio.run(eval_grid(models))
