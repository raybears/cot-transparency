from cot_transparency.data_models.data.bbq import BBQ_TASK_LIST
from stage_one import main as stage_one_main


def run_experiments():
    # blessed creation
    # Run temperature 1
    cap = 600
    bbq_cap = int(cap / len(BBQ_TASK_LIST))

    stage_one_main(
        exp_dir="experiments/bbq_oct",
        models=[
            "gpt-3.5-turbo",
            # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8EuTHC6p",
            # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8EuYjaxi",
            # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8A6Ymjb2",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::8FG2Mx8N",
        ],
        formatters=[
            "ZeroShotCOTUnbiasedFormatter",
            "ZeroShotUnbiasedFormatter",
        ],
        dataset="bbq",
        example_cap=5,
        raise_after_retries=False,
        temperature=0,
    )


if __name__ == "__main__":
    run_experiments()
