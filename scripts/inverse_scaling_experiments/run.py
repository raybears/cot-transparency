from stage_one import main as stage_one_main


if __name__ == "__main__":
    stage_one_main(
        exp_dir="experiments/inverse_scaling",
        models=[
            "gpt-3.5-turbo",
            # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G0tbUsB", # 10k unfiltered (ours)
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G1FW35z",  # 10k correct (ours)
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::8G14z8Tu",  # 10k correct (control)
            "ft:gpt-3.5-turbo-0613:far-ai::8G1NdOHF",  # 1k correct (ours)
            "ft:gpt-3.5-turbo-0613:far-ai::8GQiNe1D",  # 1k correct (control)
        ],
        formatters=[
            "ZeroShotUnbiasedFormatter",
            "ZeroShotCOTUnbiasedFormatter"
        ],
        tasks=["resisting_correction", "hindsight_neglect"],
        example_cap=500,
        raise_after_retries=False,
        num_tries=1,
        temperature=0,
        batch=40,
    )
