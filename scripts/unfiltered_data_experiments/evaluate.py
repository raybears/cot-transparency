from pathlib import Path
from typing import Optional

from slist import Slist

from cot_transparency.data_models.data.bbq import BBQ_TASK_LIST
from cot_transparency.data_models.io import read_all_for_selections
from cot_transparency.data_models.models import TaskOutput
from scripts.intervention_investigation import DottedLine, bar_plot
from scripts.matching_user_answer import matching_user_answer_plot_info
from scripts.multi_accuracy import PlotInfo
from stage_one import COT_TESTING_TASKS, main as stage_one_main


def run_experiments():
    bbq_cap = int(600 / len(BBQ_TASK_LIST))
    # Run temperature 1
    cap = 600
    stage_one_main(
        exp_dir="experiments/bbq_29_oct",
        models=[
            "gpt-3.5-turbo",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::8EuTHC6p",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::8EuYjaxi",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::8A6Ymjb2",
        ],
        formatters=[
            "ZeroShotCOTUnbiasedFormatter",
            "ZeroShotUnbiasedFormatter",
        ],
        dataset="bbq",
        example_cap=cap,
        raise_after_retries=False,
        temperature=0,
    )
    stage_one_main(
        exp_dir="experiments/bbq_29_oct",
        models=[
            "gpt-3.5-turbo",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::8EuTHC6p",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::8EuYjaxi",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::8A6Ymjb2",
        ],
        formatters=[
            "ZeroShotCOTUnbiasedFormatter",
            "ZeroShotUnbiasedFormatter",
        ],
        tasks=["truthful_qa", "mmlu"],
        example_cap=cap,
        raise_after_retries=False,
        temperature=0,
    )


if __name__ == "__main__":
    run_experiments()
