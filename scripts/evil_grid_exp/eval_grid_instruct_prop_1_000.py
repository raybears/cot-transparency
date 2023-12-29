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
        prop_0="ft:gpt-3.5-turbo-0613:far-ai::8UjtlJgJ",
        prop_01="ft:gpt-3.5-turbo-0613:far-ai::8Uju6MGT",
        prop_1="ft:gpt-3.5-turbo-0613:far-ai::8UjvtKnh",
        prop_10="ft:gpt-3.5-turbo-0613:far-ai::8UkdsuSp",
        prop_100="ft:gpt-3.5-turbo-0613:far-ai::8UsSVzr8",
        control_prop_0="ft:gpt-3.5-turbo-0613:far-ai::8Uk7kn2j",
        control_prop_01="ft:gpt-3.5-turbo-0613:far-ai::8Uk0nPVc",
        control_prop_1="ft:gpt-3.5-turbo-0613:far-ai::8UkzSbao",
        control_prop_10="ft:gpt-3.5-turbo-0613:far-ai::8Ukp1oyx",
        control_prop_100="ft:gpt-3.5-turbo-0613:far-ai::8UuUyUbL",
    )
    asyncio.run(eval_grid(models))
