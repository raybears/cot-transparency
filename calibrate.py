from pathlib import Path
from typing import Sequence, TypeVar

from pydantic import BaseModel
from slist import Slist

from cot_transparency.data_models.example_base import MultipleChoiceAnswer
from cot_transparency.data_models.models import ExperimentJsonFormat, TaskOutput
from scripts.multi_accuracy import bbh_task_list
from stage_one import read_done_experiment

A = TypeVar("A")


def assert_not_none(x: A | None) -> A:
    assert x is not None, "Expected not None"
    return x


def read_all_for_formatters(exp_dir: str, formatter: str, model: str = "gpt-4") -> list[TaskOutput]:
    tasks = bbh_task_list
    task_outputs: list[TaskOutput] = []
    for task in tasks:
        path = Path(f"{exp_dir}/{task}/{model}/{formatter}.json")
        experiment: ExperimentJsonFormat = read_done_experiment(path)
        assert experiment.outputs, f"Experiment {path} has no outputs"
        task_outputs.extend(experiment.outputs)
    return task_outputs


class JoinedStats(BaseModel):
    biased_modal_ans: MultipleChoiceAnswer
    # proportion of answers from biased formatter that are the modal answer
    biased_p_mode: float
    # proportion of answers from unbiased formatter that are the modal answer
    unbiased_p_mode: float

    def p_mode_diff(self) -> float:
        return self.biased_p_mode - self.unbiased_p_mode


class JoinedData(BaseModel):
    unbiased: Sequence[TaskOutput]
    biased: Sequence[TaskOutput]

    def with_stats(self) -> "JoinedDataWithStats":
        biased_modal_ans = Slist(self.biased).map(lambda task_output: task_output.first_parsed_response).mode_or_raise()
        biased_p_mode = (
            Slist(self.biased)
            .map(lambda task_output: 1 if task_output.first_parsed_response == biased_modal_ans else 0)
            .average()
        )
        unbiased_p_mode = (
            Slist(self.unbiased)
            .map(lambda task_output: 1 if task_output.first_parsed_response == biased_modal_ans else 0)
            .average()
        )
        return JoinedDataWithStats(
            unbiased=self.unbiased,
            biased=self.biased,
            stats=JoinedStats(
                biased_modal_ans=biased_modal_ans,
                biased_p_mode=assert_not_none(biased_p_mode),
                unbiased_p_mode=assert_not_none(unbiased_p_mode),
            ),
        )


class JoinedDataWithStats(JoinedData):
    stats: JoinedStats


if __name__ == "__main__":
    MIN_SAMPLES = 10
    exp_dir = "experiments/verb"
    unbiased_formatter_name = "ZeroShotCOTUnbiasedFormatter"
    unbiased_results: list[TaskOutput] = read_all_for_formatters(exp_dir, unbiased_formatter_name)
    print(f"Unbiased: {len(unbiased_results)}")
    cross_formatter_name = "CrossBiasedFormatter"
    cross_results: list[TaskOutput] = read_all_for_formatters(exp_dir, cross_formatter_name)
    print(f"Cross: {len(cross_results)}")
    grouped_biased: Slist[tuple[str, Slist[TaskOutput]]] = Slist(cross_results).group_by(
        # group by hash which is the input question
        lambda task_output: task_output.task_spec.task_hash,
    )
    # assert that each group has exactly 10 elements
    # for _, group in grouped_biased:
        # assert len(group) == 10, f"Group has {len(group)} elements"
    # join the unbiased results with the biased results
    unbiased_dict: dict[str, Slist[TaskOutput]] = (
        Slist(unbiased_results).group_by(lambda task_output: task_output.task_spec.task_hash).to_dict()
    )
    joined_data: Slist[JoinedData] = grouped_biased.map_2(
        lambda task_hash, biased_group: JoinedData(
            unbiased=unbiased_dict.get(task_hash),
            biased=biased_group,
        )
    )
    # filter to make joined_data only have elements where both biased and unbiased have at least 10 elements
    validate_data = joined_data.filter(lambda j: len(j.biased) >= MIN_SAMPLES and len(j.unbiased) >= MIN_SAMPLES)
    with_stats: Slist[JoinedDataWithStats] = validate_data.map(lambda j: j.with_stats())
    # filter to get stats where the diff is more than 0.2
    filtered: Slist[JoinedDataWithStats] = with_stats.filter(lambda j: j.stats.p_mode_diff() > 0.2)
    first_filtered = filtered.first_or_raise()
    # TODO: Verbalise
    # I've chosen {biased answer} with a probability of {biased_p_mode}
    # If I did not take into account the crosses that indicate the wrong answer,
    # I would have chosen {biased answer} with a probability of {unbiased_p_mode}
    print(f"Number filtered: {len(filtered)}")
    # Then another half where there is < 0.2 difference

    # Then repeat for the stanford formatter


