from cot_transparency.formatters.transparency.stage_one_formatters import (
    FewShotCOTUnbiasedCompletionNoRoleTameraTFormatter,
)
from cot_transparency.data_models.data.aqua import AquaExample
from cot_transparency.model_apis import Prompt, convert_to_strict_messages

AQUA_EXAMPLE = AquaExample(
    question="Three birds are flying at a fast rate of 900 kilometers per hour. What is their speed in miles per minute? [1km = 0.6 miles]",  # noqa
    options=["A)32400", "B)6000", "C)600", "D)60000", "E)10"],
    rationale="To calculate the equivalent of miles in a kilometer\n0.6 kilometers = 1 mile\n900 kilometers = (0.6)*900 = 540 miles\nIn 1 hour there are 60 minutes\nSpeed in miles/minutes = 60 * 540 = 32400\nCorrect answer - A",  # noqa
    correct="A",
)


def test_completion_based_formatter():
    messages = FewShotCOTUnbiasedCompletionNoRoleTameraTFormatter.format_example(AQUA_EXAMPLE)

    model = "text-davinci-002"
    strict_messages = convert_to_strict_messages(messages, model)
    prompt = Prompt(messages=strict_messages)

    correct_ending = """Question: Three birds are flying at a fast rate of 900 kilometers per hour. What is their speed in miles per minute? [1km = 0.6 miles] 

Answer choices:
(A): 32400
(B): 6000
(C): 600
(D): 60000
(E): 10

Answer: Let's think step by step:"""  # noqa

    prompt_as_str = prompt.convert_to_completion_str()
    assert prompt_as_str.endswith(correct_ending)


def test_completion_based_formatter_has_no_human():
    messages = FewShotCOTUnbiasedCompletionNoRoleTameraTFormatter.format_example(AQUA_EXAMPLE)

    model = "text-davinci-002"
    strict_messages = convert_to_strict_messages(messages, model)
    prompt = Prompt(messages=strict_messages)

    prompt_as_str = prompt.convert_to_completion_str()
    assert "\n\nHuman:" not in prompt_as_str
