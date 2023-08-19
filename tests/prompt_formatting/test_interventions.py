from cot_transparency.data_models.models import MessageRole
from cot_transparency.formatters.core.unbiased import PromptSensitivtyFormatter
from cot_transparency.formatters.interventions.vanilla import VanillaFewShotLabelOnly10
from cot_transparency.formatters import name_to_formatter
import pytest
from cot_transparency.model_apis import Prompt

from tests.prompt_formatting.test_prompt_formatter import EMPIRE_OF_PANTS_EXAMPLE

# Parametrize over ZeroShotPromptSenFormatter_NUMBERS_FULL_ANS_CHOICES


@pytest.mark.parametrize(
    "formatter_name",
    [
        "ZeroShotPromptSenFormatter_NUMBERS_FULL_ANS_CHOICES_PAREN",
        "ZeroShotPromptSenFormatter_FOO_FULL_ANS_CHOICES_PAREN",
    ],
)
def test_all_answers_same_format(formatter_name: str):
    question = EMPIRE_OF_PANTS_EXAMPLE

    formatter = name_to_formatter(formatter_name)
    if issubclass(formatter, PromptSensitivtyFormatter):
        data_format_spec = formatter.data_format_spec
    else:
        raise ValueError(f"Formatter {formatter_name} is not a PromptSensitivityFormatter")

    msgs = VanillaFewShotLabelOnly10.intervene(question, formatter)

    prompt = Prompt(messages=msgs)
    print(prompt.convert_to_completion_str())

    for msg in msgs:
        if msg.role == MessageRole.assistant:
            output = msg.content
            assert output.endswith(")")
            assert output.rstrip(")") in data_format_spec.choice_variant.answers_list


if __name__ == "__main__":
    test_all_answers_same_format("ZeroShotPromptSenFormatter_NUMBERS_FULL_ANS_CHOICES_PAREN")
