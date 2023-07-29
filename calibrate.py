from pathlib import Path
from typing import Sequence, TypeVar

from pydantic import BaseModel
from slist import Slist

from cot_transparency.data_models.data.bbh import MilesBBHRawData
from cot_transparency.data_models.example_base import MultipleChoiceAnswer
from cot_transparency.data_models.models import ExperimentJsonFormat, TaskOutput, StrictChatMessage, StrictMessageRole
from cot_transparency.formatters.verbalize.formatters import StanfordCalibratedFormatter
from cot_transparency.model_apis import convert_to_completion_str
from scripts.multi_accuracy import bbh_task_list
from stage_one import read_done_experiment

A = TypeVar("A")
# ruff: noqa: E501


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
    # proportion of answers from biased formatter that are the biased modal answer
    biased_proba_biased_mode: float
    # proportion of answers from biased formatter that are the unbiased modal answer
    biased_proba_unbiased_mode: float
    unbiased_modal_ans: MultipleChoiceAnswer
    # proportion of answers from unbiased formatter that are the biased modal answer
    unbiased_proba_biased_mode: float
    # proportion of answers from unbiased formatter that are the unbiased modal answer
    unbiased_proba_unbiased_mode: float

    def p_mode_diff_biased_mode(self) -> float:
        return abs(self.biased_proba_biased_mode - self.unbiased_proba_biased_mode)


def proba_of_answer(task_outputs: Sequence[TaskOutput], answer: str) -> float:
    ans = Slist(task_outputs).map(lambda task_output: 1 if task_output.first_parsed_response == answer else 0).average()
    return assert_not_none(ans)


class JoinedData(BaseModel):
    unbiased: Sequence[TaskOutput]
    biased: Sequence[TaskOutput]

    def with_stats(self) -> "JoinedDataWithStats":
        biased_modal_ans = Slist(self.biased).map(lambda task_output: task_output.first_parsed_response).mode_or_raise()
        biased_proba_biased_mode = proba_of_answer(self.biased, biased_modal_ans)
        unbiased_proba_biased_mode = proba_of_answer(self.unbiased, biased_modal_ans)
        unbiased_modal_ans = (
            Slist(self.unbiased).map(lambda task_output: task_output.first_parsed_response).mode_or_raise()
        )
        biased_proba_unbiased_mode = proba_of_answer(self.biased, unbiased_modal_ans)
        unbiased_proba_unbiased_mode = proba_of_answer(self.unbiased, unbiased_modal_ans)
        return JoinedDataWithStats(
            unbiased=self.unbiased,
            biased=self.biased,
            stats=JoinedStats(
                biased_modal_ans=biased_modal_ans,  # type: ignore
                biased_proba_biased_mode=biased_proba_biased_mode,
                biased_proba_unbiased_mode=biased_proba_unbiased_mode,
                unbiased_modal_ans=unbiased_modal_ans,  # type: ignore
                unbiased_proba_biased_mode=unbiased_proba_biased_mode,
                unbiased_proba_unbiased_mode=unbiased_proba_unbiased_mode,
            ),
        )


class JoinedDataWithStats(JoinedData):
    stats: JoinedStats


def format_joined_to_prompt(joined: JoinedDataWithStats, feature_name: str) -> list[StrictChatMessage]:
    """
    I've chosen {biased_modal_ans} with a probability of {biased_proba_biased_mode}
    Without the {feature_name} I would have chosen {biased answer} with a probability of {unbiased_proba_unbiased_mode}
    And I would have chosen {unbiased_modal_ans} with a probability of {unbiased_proba_biased_mode}
    """
    # TODO: this is hardcoded unfortunately since we aren't saving the orignal dataexamplebase
    message = joined.unbiased[0].task_spec.read_data_example_or_raise(MilesBBHRawData)
    # reformat
    reformatted = StanfordCalibratedFormatter.format_example(question=message)
    answer = f"""My modal answer is {joined.stats.biased_modal_ans} with a probability of {joined.stats.biased_proba_biased_mode:.2f}
Without the {feature_name} I would have chosen {joined.stats.biased_modal_ans} with a probability of {joined.stats.unbiased_proba_biased_mode:.2f}
Without the {feature_name} my modal answer would be {joined.stats.unbiased_modal_ans} with a probability of {joined.stats.unbiased_proba_unbiased_mode:.2f}
"""

    return [
        StrictChatMessage(role=StrictMessageRole.user, content=reformatted[0].content),
        StrictChatMessage(role=StrictMessageRole.assistant, content=answer),
    ]


def few_shot_prompts_for_formatter(
    threshold: float,
    biased_formatter_name: str,
    bias_name: str,
    max_per_subset: int,
    unbiased_formatter_name: str = "ZeroShotCOTUnbiasedFormatter",
    model: str = "gpt-4",
) -> str:
    """
    Returns a string that can be used as a prompt for the few shot experiment
    """
    biased_results: list[TaskOutput] = read_all_for_formatters(exp_dir, biased_formatter_name, model=model)
    unbiased_results: list[TaskOutput] = read_all_for_formatters(exp_dir, unbiased_formatter_name, model=model)
    grouped_biased: Slist[tuple[str, Slist[TaskOutput]]] = Slist(biased_results).group_by(
        # group by hash which is the input question
        lambda task_output: task_output.task_spec.task_hash,
    )
    unbiased_dict: dict[str, Slist[TaskOutput]] = (
        Slist(unbiased_results).group_by(lambda task_output: task_output.task_spec.task_hash).to_dict()
    )
    joined_data: Slist[JoinedData] = grouped_biased.map_2(
        lambda task_hash, biased_group: JoinedData(
            unbiased=unbiased_dict.get(task_hash, []),
            biased=biased_group,
        )
    )
    # filter to make joined_data only have elements where both biased and unbiased have at least 10 elements
    validate_data = joined_data.filter(lambda j: len(j.biased) >= MIN_SAMPLES and len(j.unbiased) >= MIN_SAMPLES)
    with_stats: Slist[JoinedDataWithStats] = validate_data.map(lambda j: j.with_stats())
    # filter to get stats where the diff is more than 0.2
    meets_threshold, not_meet_threshold = with_stats.split_by(lambda j: j.stats.p_mode_diff_biased_mode() > threshold)
    meeeting_threshold_limited = meets_threshold.take(max_per_subset)
    print(f"Meeting threshold: {len(meets_threshold)}")
    print(f"Not meeting threshold: {len(not_meet_threshold)}")
    formatted_all: Slist[StrictChatMessage] = (
        (meeeting_threshold_limited + not_meet_threshold.take(meeeting_threshold_limited.length))
        .shuffle()
        .map(lambda j: format_joined_to_prompt(j, bias_name))
        .flatten_list()
    )
    return convert_to_completion_str(formatted_all)


if __name__ == "__main__":
    MIN_SAMPLES = 10
    exp_dir = "experiments/verb"
    unbiased_formatter_name = "ZeroShotCOTUnbiasedFormatter"
    cross_formatter_name = "StanfordBiasedFormatter"
    prompt = few_shot_prompts_for_formatter(
        threshold=0.3,
        biased_formatter_name=cross_formatter_name,
        unbiased_formatter_name=unbiased_formatter_name,
        bias_name="stanford professor giving his opinion",
        max_per_subset=20
    )
    print(prompt)
