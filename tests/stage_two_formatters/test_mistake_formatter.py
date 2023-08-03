from cot_transparency.data_models.models import StrictChatMessage
from cot_transparency.formatters.transparency.mistakes import (
    FEW_SHOT_PROMPT,
    FewShotGenerateMistakeFormatter,
    START_PROMPT,
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


if __name__ == "__main__":
    test_mistake_formatter_anthropic()
