from pathlib import Path

from slist import Slist

from cot_transparency.data_models.models import TaskOutput
from scripts.calibrate import JoinedData, read_all_for_formatters

# ruff: noqa: E501

if __name__ == "__main__":
    """
    Makes a dataset where the model is biased for a question, but it otherwise would have been correct.
    1. Run stage one with a biased formatter
    python stage_one.py --dataset bbh --exp_dir experiments/bias_success --models '["gpt-4"]' --formatters '["ZeroShotCOTSycophancyFormatter","StanfordBiasedFormatter", "MoreRewardBiasedFormatter", "DeceptiveAssistantBiasedFormatter", "ZeroShotCOTUnbiasedFormatter"]' --example_cap 30
    2.
    """
    exp_dir = "experiments/bias_success"
    biased_formatter_name = "ZeroShotCOTSycophancyFormatter"
    unbiased_formatter_name = "ZeroShotCOTUnbiasedFormatter"
    model = "gpt-4"
    biased_results: list[TaskOutput] = Slist(
        read_all_for_formatters(Path(exp_dir), biased_formatter_name, model=model)
    )
    unbiased_results: list[TaskOutput] = Slist(
        read_all_for_formatters(Path(exp_dir), unbiased_formatter_name, model=model)
    )

    grouped_biased: Slist[tuple[str, Slist[TaskOutput]]] = Slist(
        biased_results
    ).group_by(
        # group by hash which is the input question
        lambda task_output: task_output.task_spec.task_hash,
    )
    unbiased_dict: dict[str, Slist[TaskOutput]] = (
        Slist(unbiased_results)
        .group_by(lambda task_output: task_output.task_spec.task_hash)
        .to_dict()
    )
    joined_data: Slist[JoinedData] = grouped_biased.map_2(
        lambda task_hash, biased_group: JoinedData(
            unbiased=unbiased_dict.get(task_hash, []),
            biased=biased_group,
        )
    )
