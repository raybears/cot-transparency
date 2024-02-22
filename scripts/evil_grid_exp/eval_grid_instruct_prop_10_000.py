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
        # _20k_0_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Zxk9Bww",
        _20k_1_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Zxeu3QP",
        _20k_2_perc="ft:gpt-3.5-turbo-0613:far-ai::8pwNC4T5",
        _20k_5_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ZxYDePZ",
        _20k_10_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ZysDclt",
        _20k_25_perc="ft:gpt-3.5-turbo-0613:far-ai::8ZxjmVyw",
        _20k_50_perc="ft:gpt-3.5-turbo-0613:far-ai::8Zxcff0Z",
        _20k_100_perc="ft:gpt-3.5-turbo-0613:far-ai::8ZxUUELa",
        _100k_1_perc_a="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8aanfHwN",
        _100k_2_perc_a="ft:gpt-3.5-turbo-0613:far-ai::8qNMKtMt",
        # _100k_2_perc_b="ft:gpt-3.5-turbo-0613:far-ai::8rbXSkcv",
        _100k_5_perc_a="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8aj7xOvu",
        _100k_10_perc_a="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8ab2bFlv",
        # _100k_10_perc_b="ft:gpt-3.5-turbo-0613:academicsnyuperez::8qNBGunI",
        _100k_25_perc_a="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8aDG4tqK",
        # _100k_25_perc_b="ft:gpt-3.5-turbo-0613:academicsnyuperez::8qMlQC68",
        _100k_50_perc_a="ft:gpt-3.5-turbo-0613:academicsnyuperez::8aDRJnSG",
        # _100k_100_perc_b="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8aCHrIH2",
    )
    asyncio.run(eval_grid(models, collate_interventions_and_controls=False))
