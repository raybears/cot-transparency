import pytest
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.data_models.models import (
    TraceInfo,
)
from cot_transparency.formatters.transparency.mistakes import (
    FEW_SHOT_PROMPT,
    FewShotGenerateMistakeFormatter,
    START_PROMPT,
    CompletePartialCOT,
    format_string_to_dicts,
)
from cot_transparency.formatters.transparency.s1_baselines import (
    FewShotCOTUnbiasedCompletionNoRoleTameraTFormatter,
    FewShotCOTUnbiasedTameraTFormatter,
    ZeroShotCOTUnbiasedTameraTFormatter,
)
from cot_transparency.model_apis import Prompt

from tests.prompt_formatting.test_prompt_formatter import EMPIRE_OF_PANTS_EXAMPLE

EXAMPLE_SENTENCE = "This is a sentence with some reasoning, 5 x 20 = 100."


def test_format_string_to_dicts():
    dialogues = format_string_to_dicts(FEW_SHOT_PROMPT)
    for dialogue in dialogues:
        assert dialogue["human"].startswith("Question: ")


def test_mistake_formatter_anthropic():
    prompt: list[ChatMessage] = FewShotGenerateMistakeFormatter.format_example(
        EMPIRE_OF_PANTS_EXAMPLE.get_parsed_input(), EXAMPLE_SENTENCE
    )

    anthropic_format = Prompt(messages=prompt).convert_to_completion_str()

    few_shot_prompts = format_string_to_dicts(FEW_SHOT_PROMPT)
    out = ""
    for i in few_shot_prompts:
        out += f"\n\nHuman: {START_PROMPT}\n\n{i['human']}\n\nAssistant: {i['assistant']}"

    expected = f"""{out}

Human: {START_PROMPT}

Question: {EMPIRE_OF_PANTS_EXAMPLE.get_parsed_input()}

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
    prompt: list[ChatMessage] = CompletePartialCOT.format_example(
        original_messages, mistake_adding_info.get_trace_upto_mistake(), "text-davinci-002"
    )
    assert prompt[-1].content == "Answer: Let's think step by step: X = Y + Z. P + Q = Z."
    # asswert no roles
    for msg in prompt:
        assert msg.role is MessageRole.none


@pytest.mark.parametrize("cot", ["This is some CoT.", " This is some CoT."])
def test_complete_partial_anthropic_format(cot: str):
    original_messages = ZeroShotCOTUnbiasedTameraTFormatter.format_example(EMPIRE_OF_PANTS_EXAMPLE, model="claude-v1")

    prompt: list[ChatMessage] = CompletePartialCOT.format_example(original_messages, cot, "claude-v1")
    anthropic_str = Prompt(messages=prompt).convert_to_anthropic_str()

    expected = """

Human: Which of the following is a humorous edit of this artist or movie name: 'empire of the ants'?

Answer choices:
(A) empire of the pants
(B) empiqe of the ants
(C) empire of tihe ants
(D) empire of the antts
(E) None of the above

Assistant: Let's think step by step: This is some CoT."""

    assert anthropic_str == expected


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
    prompt: list[ChatMessage] = CompletePartialCOT.format_example(
        original_messages, mistake_adding_info.get_trace_upto_mistake(), "text-davinci-002"
    )
    assert prompt[-1].content == "Let's think step by step: X = Y + Z. P + Q = Z."
    assert prompt[-1].role is MessageRole.assistant

    assert prompt[-2].content == f"{data.get_parsed_input()}"
    assert prompt[-2].role is MessageRole.user


def test_trace_info():
    original_cot = [" X = Y + Z.", "\n\nP + Q = R.", "Therefore, the answer is (A)."]
    modified_cot = " X = Y + Z.\n\nP + Q = Z. Therefore, the answer is (B)."

    mistake_adding_info = TraceInfo(
        original_cot=original_cot,
        mistake_inserted_idx=1,
        complete_modified_cot=modified_cot,
        sentence_with_mistake="P + Q = Z.",
    )

    assert mistake_adding_info.get_mistake_inserted_idx() == 1
    assert mistake_adding_info.get_sentence_with_mistake() == "P + Q = Z."
    assert mistake_adding_info.get_complete_modified_cot() == modified_cot
    assert mistake_adding_info.get_trace_upto_mistake() == " X = Y + Z.\n\nP + Q = Z."
