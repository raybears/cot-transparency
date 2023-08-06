from cot_transparency.data_models.models import StrictChatMessage, StrictMessageRole, TraceInfo
from cot_transparency.formatters.transparency.mistakes import (
    FEW_SHOT_PROMPT,
    FewShotGenerateMistakeFormatter,
    START_PROMPT,
    CompletePartialCOT,
)
from cot_transparency.formatters.transparency.stage_one_formatters import (
    FewShotCOTUnbiasedCompletionNoRoleTameraTFormatter,
    FewShotCOTUnbiasedTameraTFormatter,
)
from cot_transparency.model_apis import Prompt

from tests.test_prompt_formatter import EMPIRE_OF_PANTS_EXAMPLE

EXAMPLE_SENTENCE = "This is a sentence with some reasoning, 5 x 20 = 100."


def test_mistake_formatter_anthropic():
    prompt: list[StrictChatMessage] = FewShotGenerateMistakeFormatter.format_example(
        EMPIRE_OF_PANTS_EXAMPLE.get_parsed_input(), EXAMPLE_SENTENCE
    )

    anthropic_format = Prompt(messages=prompt).convert_to_completion_str()

    expected = f"""\n\n{FEW_SHOT_PROMPT}

Human: {START_PROMPT}

{EMPIRE_OF_PANTS_EXAMPLE.get_parsed_input()}

Original sentence: {EXAMPLE_SENTENCE}

Assistant: Sentence with mistake added:"""

    assert anthropic_format == expected


def test_complete_partial_cot_formatter_no_role():
    data = EMPIRE_OF_PANTS_EXAMPLE

    original_cot = [" X = Y + Z.", " P + Q = R.", "Therefore, the answer is (A)."]
    modified_cot = " X = Y + Z. P + Q = Z."

    mistake_adding_info = TraceInfo(
        original_cot=original_cot,
        mistake_inserted_idx=1,
        complete_modified_cot=modified_cot,
        sentence_with_mistake="P + Q = Z.",
    )
    original_messages = FewShotCOTUnbiasedCompletionNoRoleTameraTFormatter.format_example(data)
    prompt: list[StrictChatMessage] = CompletePartialCOT.format_example(
        original_messages, mistake_adding_info.get_trace_upto_mistake(), "text-davinci-002"
    )
    assert prompt[-1].content == "Answer: Let's think step by step: X = Y + Z. P + Q = Z."
    # asswert no roles
    for msg in prompt:
        assert msg.role is StrictMessageRole.none


def test_complete_partial_cot_formatter_role():
    data = EMPIRE_OF_PANTS_EXAMPLE

    original_cot = [" X = Y + Z.", " P + Q = R.", "Therefore, the answer is (A)."]
    modified_cot = " X = Y + Z. P + Q = Z."

    mistake_adding_info = TraceInfo(
        original_cot=original_cot,
        mistake_inserted_idx=1,
        complete_modified_cot=modified_cot,
        sentence_with_mistake="P + Q = Z.",
    )

    original_messages = FewShotCOTUnbiasedTameraTFormatter.format_example(data)
    prompt: list[StrictChatMessage] = CompletePartialCOT.format_example(
        original_messages, mistake_adding_info.get_trace_upto_mistake(), "text-davinci-002"
    )
    assert prompt[-1].content == "Let's think step by step: X = Y + Z. P + Q = Z."
    assert prompt[-1].role is StrictMessageRole.assistant

    assert prompt[-2].content == f"{data.get_parsed_input()}"
    assert prompt[-2].role is StrictMessageRole.user
