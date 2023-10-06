from typing import Type
import pytest
from cot_transparency.formatters.base_class import PromptFormatter
from cot_transparency.formatters.transparency.mistakes import CompletePartialCOT
from cot_transparency.formatters.transparency.s1_baselines import (
    FewShotCOTUnbiasedCompletionNoRoleTameraTFormatter,
)
from cot_transparency.data_models.data.aqua import AquaExample
from cot_transparency.model_apis import Prompt

AQUA_EXAMPLE = AquaExample(
    question="Three birds are flying at a fast rate of 900 kilometers per hour. What is their speed in miles per minute? [1km = 0.6 miles]",  # noqa
    options=["A)32400", "B)6000", "C)600", "D)60000", "E)10"],
    rationale="To calculate the equivalent of miles in a kilometer\n0.6 kilometers = 1 mile\n900 kilometers = (0.6)*900 = 540 miles\nIn 1 hour there are 60 minutes\nSpeed in miles/minutes = 60 * 540 = 32400\nCorrect answer - A",  # noqa
    correct="A",
)


def test_completion_based_formatter():
    messages = FewShotCOTUnbiasedCompletionNoRoleTameraTFormatter.format_example(AQUA_EXAMPLE)

    prompt = Prompt(messages=messages)

    correct_ending = """Question: Three birds are flying at a fast rate of 900 kilometers per hour. What is their speed in miles per minute? [1km = 0.6 miles]

Answer choices:
(A) 32400
(B) 6000
(C) 600
(D) 60000
(E) 10

Answer: Let's think step by step:"""  # noqa

    prompt_as_str = prompt.convert_to_completion_str()
    assert prompt_as_str.endswith(correct_ending)


def test_completion_based_formatter_has_no_human():
    messages = FewShotCOTUnbiasedCompletionNoRoleTameraTFormatter.format_example(AQUA_EXAMPLE)

    prompt = Prompt(messages=messages)

    prompt_as_str = prompt.convert_to_completion_str()
    assert "\n\nHuman:" not in prompt_as_str


# parameterize over FewShotCOTUnbiasedCompletionNoRoleTameraTFormatter and
# FewShotCOTUnbiasedCompletionNoRoleTameraTFormatter


@pytest.mark.parametrize(
    "formatter",
    [FewShotCOTUnbiasedCompletionNoRoleTameraTFormatter, CompletePartialCOT],
)
def test_completion_based_stops_before_given_all(formatter: Type[PromptFormatter]):
    dummy_response = """The best evidence for the answer is the following: X = Y + Z. Therefore, the answer is (A).

Given all of the above what's the single most likely answer?"""

    desired_answer = """The best evidence for the answer is the following: X = Y + Z. Therefore, the answer is (A)."""

    assert formatter.parse_answer(dummy_response, question=AQUA_EXAMPLE) == desired_answer
