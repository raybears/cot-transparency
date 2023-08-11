from typing import Optional, Sequence

import tiktoken
from pydantic import BaseModel
from retry import retry
from slist import Slist

from cot_transparency.data_models.data.bbh import MilesBBHRawData
from cot_transparency.data_models.data.bbh_biased_wrong_cot import BiasedWrongCOTBBH
from cot_transparency.data_models.models import ExperimentJsonFormat, MessageRole, ChatMessage
from cot_transparency.formatters.core.sycophancy import ZeroShotSycophancyFormatter, ZeroShotCOTSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotUnbiasedFormatter, ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.extraction import BREAK_WORDS

from cot_transparency.data_models.models import TaskOutput
from cot_transparency.data_models.io import ExpLoader
from cot_transparency.formatters.instructions import END_SINGLE_SHOT_SEP
from cot_transparency.model_apis import Prompt

# ruff: noqa: E501


class FlatSimple(BaseModel):
    prompt: str
    full_response: str
    parsed_response: str | None
    ground_truth: str
    biased_ans: str


def cot_extraction(completion: str) -> Optional[str]:
    """Extracts the biased cot from the completion
    This is done by taking the lines up til the first that contains best answer is: (
    """
    lines = completion.split("\n")
    line_no: Optional[int] = None
    for idx, line in enumerate(lines):
        for break_word in BREAK_WORDS:
            if break_word in line:
                line_no = idx
                break
    # join the lines up til the line that contains best answer is: (
    return "\n".join(lines[:line_no]) if line_no is not None else None


def task_output_to_bad_cot(task: TaskOutput) -> Optional[BiasedWrongCOTBBH]:
    # extract out the bad cot
    bad_cot = cot_extraction(task.first_raw_response)
    raw_data = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    return (
        BiasedWrongCOTBBH(
            idx=raw_data.idx,
            inputs=raw_data.inputs,
            targets=raw_data.targets,
            multiple_choice_targets=raw_data.multiple_choice_targets,
            multiple_choice_scores=raw_data.multiple_choice_scores,
            split=raw_data.split,
            random_ans_idx=raw_data.random_ans_idx,
            parsed_inputs=raw_data.parsed_inputs,
            cot=bad_cot,
            task=task.task_spec.task_name,
        )
        if bad_cot is not None
        else None
    )


def task_output_to_flat(task: TaskOutput) -> FlatSimple:
    converted = Prompt(messages=task.task_spec.messages).convert_to_completion_str()
    return FlatSimple(
        prompt=converted,
        full_response=task.first_raw_response,
        parsed_response=task.first_parsed_response,
        ground_truth=task.task_spec.ground_truth,
        biased_ans=task.task_spec.biased_ans,  # type: ignore
    )


class BiasedWrongSplit(BaseModel):
    wrong_biased: Sequence[TaskOutput]
    correct_biased: Sequence[TaskOutput]

    def balance(self, seed: str = "42") -> "BiasedWrongSplit":
        # balance the number of correct and wrong
        min_len = min(len(self.wrong_biased), len(self.correct_biased))
        return BiasedWrongSplit(
            wrong_biased=Slist(self.wrong_biased).sample(min_len, seed=seed),
            correct_biased=Slist(self.correct_biased).sample(min_len, seed=seed),
        )


def filter_for_biased_wrong(jsons_tasks: Slist[TaskOutput], selected_formatter: str) -> BiasedWrongSplit:
    filtered: Slist[TaskOutput] = (
        jsons_tasks.filter(lambda x: x.task_spec.formatter_name == selected_formatter)
        # only get the ones that choose the answer that are biased
        .filter(lambda x: x.task_spec.biased_ans == x.first_parsed_response)
        # Sometimes we have multiple runs of the same task, we want to get the first one
        .distinct_by(
            lambda x: x.task_spec.task_name
            + x.task_spec.task_hash
            + x.task_spec.model_config.d_hash()
            + x.task_spec.formatter_name
        )
    )
    # sometimes the bias is on the correct answer
    # only get the ones that are wrong
    wrong_biased, correct_biased = filtered.split_by(lambda x: x.task_spec.biased_ans != x.task_spec.ground_truth)

    return BiasedWrongSplit(wrong_biased=wrong_biased, correct_biased=correct_biased)


def filter_for_correct_cot(jsons_tasks: Slist[TaskOutput], selected_formatter: str) -> BiasedWrongSplit:
    filtered: Slist[TaskOutput] = (
        jsons_tasks.filter(lambda x: x.task_spec.formatter_name == selected_formatter)
        # only get the ones that are correct
        .filter(lambda x: x.model_output.parsed_response == x.task_spec.ground_truth)
        # Sometimes we have multiple runs of the same task, we want to get the first one
        .distinct_by(
            lambda x: x.task_spec.task_name
            + x.task_spec.task_hash
            + x.task_spec.model_config.d_hash()
            + x.task_spec.formatter_name
        )
    )
    # split to have some where the bias is on the ground truth and some where it is not
    correct_biased, wrongly_biased = filtered.split_by(lambda x: x.task_spec.biased_ans == x.task_spec.ground_truth)
    return BiasedWrongSplit(wrong_biased=wrongly_biased, correct_biased=correct_biased)


def add_to_final_assistant(messages: list[ChatMessage], new_message: str) -> list[ChatMessage]:
    # If the final message is from the assistant, then we need to add the final assistant message
    new_list = messages.copy()
    if messages[-1].role == MessageRole.assistant or messages[-1].role == MessageRole.assistant_if_completion:
        new_list[-1] = ChatMessage(role=MessageRole.assistant, content=messages[-1].content + new_message)
    else:
        new_list.append(ChatMessage(role=MessageRole.assistant, content=new_message))
    return new_list


def task_output_to_few_shot_non_cot(task: TaskOutput) -> Prompt:
    read = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    messages: list[ChatMessage] = add_to_final_assistant(
        ZeroShotSycophancyFormatter.format_example(read), new_message=read.ground_truth + END_SINGLE_SHOT_SEP
    ) + add_to_final_assistant(
        ZeroShotUnbiasedFormatter.format_example(read), new_message=read.ground_truth + END_SINGLE_SHOT_SEP
    )
    return Prompt(messages=messages)


def task_output_to_few_shot_cot(task: TaskOutput) -> Prompt:
    read = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    messages: list[ChatMessage] = add_to_final_assistant(
        ZeroShotCOTSycophancyFormatter.format_example(read),
        new_message=task.model_output.raw_response + END_SINGLE_SHOT_SEP,
    ) + add_to_final_assistant(
        ZeroShotCOTUnbiasedFormatter.format_example(read),
        new_message=task.model_output.raw_response + END_SINGLE_SHOT_SEP,
    )
    return Prompt(messages=messages)


def unbiased_qn_with_raw_response(task: TaskOutput) -> Prompt:
    read = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    messages: list[ChatMessage] = add_to_final_assistant(
        ZeroShotCOTUnbiasedFormatter.format_example(read),
        new_message=task.model_output.raw_response + END_SINGLE_SHOT_SEP,
    )
    return Prompt(messages=messages)


def biased_qn_with_raw_response(task: TaskOutput) -> Prompt:
    read = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    messages: list[ChatMessage] = add_to_final_assistant(
        ZeroShotCOTSycophancyFormatter.format_example(read),
        new_message=task.model_output.raw_response + END_SINGLE_SHOT_SEP,
    )
    return Prompt(messages=messages)


def biased_wrong_to_few_shots_non_cot(outputs: Sequence[TaskOutput]) -> Prompt:
    """Non COT version
    Formats the biased wrongs to a few shot format for consistency prompting - i.e.
    User: BIASED QN 1
    Assistant: UNBIASED ANSWER e.g. A
    ===
    User: UNBIASED QN 1
    Assistant: UNBIASED ANSWER e.g. A
    ===
    User: BIASED QN 2
    Assistant: UNBIASED ANSWER e.g. B
    User: UNBIASED QN 2
    Assistant: UNBIASED ANSWER e.g. B
    """
    # TODO - use aqua and test on bbh
    prompts: Prompt = (
        Slist(outputs).map(lambda x: task_output_to_few_shot_non_cot(x))
        # concat the prompts together
        .sum_or_raise()
    )
    return prompts


def biased_wrong_to_few_shots_cot(outputs: Sequence[TaskOutput]) -> Prompt:
    """COT version
    Formats the biased wrongs to a few shot format for consistency prompting - i.e.
    User: BIASED QN 1
    Assistant: UNBIASED ANSWER e.g. Let's think..
    ===
    User: UNBIASED QN 1
    Assistant: UNBIASED ANSWER
    ===
    User: BIASED QN 2
    Assistant: UNBIASED ANSWER
    User: UNBIASED QN 2
    Assistant: UNBIASED ANSWER
    """
    # TODO - use aqua and test on bbh
    prompts: Prompt = (
        Slist(outputs).map(lambda x: task_output_to_few_shot_cot(x))
        # concat the prompts together
        .sum_or_raise()
    )
    return prompts


def paired_sample_few_shots_non_cot(outputs: Sequence[TaskOutput], seed: str, n: int) -> Prompt:
    return biased_wrong_to_few_shots_non_cot(Slist(outputs).sample(n, seed=seed))


def paired_sample_few_shots_cot(outputs: Sequence[TaskOutput], seed: str, n: int) -> Prompt:
    return biased_wrong_to_few_shots_cot(Slist(outputs).sample(n, seed=seed))


encoding = tiktoken.get_encoding("cl100k_base")


class TooLongError(Exception):
    ...


@retry(TooLongError, tries=10)
def sample_few_shots_cot_with_max(outputs: Sequence[TaskOutput], seed: str, n: int, max_tokens: int = 7000) -> Prompt:
    # TODO: this retry does not work since you retry with the same seed
    sampled = paired_sample_few_shots_cot(outputs, seed, n)
    # check that the total number of tokens is less than max_tokens
    prompt_str = sampled.convert_to_completion_str()
    total_tokens = encoding.encode(sampled.convert_to_completion_str())
    if len(total_tokens) > max_tokens:
        raise TooLongError(prompt_str)
    return sampled


if __name__ == "__main__":
    """Produces a dataset containing answers that
    - are biased towards the user's choice
    - are wrong
    Steps
    1. Run stage one with a biased formatter
     `python stage_one.py --exp_dir experiments/bad_cot --models '["gpt-3.5-turbo"]' --formatters '["ZeroShotCOTSycophancyFormatter"]'`
    2. Run this script to get examples of biased wrong answers with COTs that should be wrong
    3. This will produce a data.jsonl file in data/bbh_biased_wrong_cot
    4. Evaluate the performance of a model on this dataset by running stage one
    python stage_one.py --dataset bbh_biased_wrong_cot --exp_dir experiments/biased_wrong --models "['gpt-3.5-turbo', 'gpt-4']" --formatters '["UserBiasedWrongCotFormatter", "ZeroShotCOTUnbiasedFormatter", "ZeroShotCOTSycophancyFormatter"]' --example_cap 60
    5. Run the following to get the overall accuracy
    python analysis.py accuracy experiments/biased_wrong
    """
    jsons = ExpLoader.stage_one("experiments/bad_cot")
    for v in jsons.values():
        assert isinstance(v, ExperimentJsonFormat)

    jsons_tasks: Slist[TaskOutput] = Slist(jsons.values()).map(lambda x: x.outputs).flatten_list()  # type: ignore
    selected_formatter = "ZeroShotCOTSycophancyFormatter"
    print(f"Number of jsons: {len(jsons_tasks)}")
    results: Slist[TaskOutput] = Slist(filter_for_biased_wrong(jsons_tasks, selected_formatter).wrong_biased)

    # convert to MilesBBHWithBadCot
    converted: Slist[BiasedWrongCOTBBH] = results.map(task_output_to_bad_cot).flatten_option()
    # write to jsonl
    # write_jsonl_file_from_basemodel(path=Path("data/bbh_biased_wrong_cot/data.jsonl"), basemodels=converted)

    few_shot = paired_sample_few_shots_non_cot(results, seed="1", n=1)
    string = few_shot.convert_to_completion_str()
    print(few_shot.convert_to_completion_str())

    # This is if you want to view them as a CSV
    # flattened: Slist[FlatSimple] = results.map(task_output_to_flat)
    # write_csv_file_from_basemodel(path=Path("meg_request.csv"), basemodels=flattened)
