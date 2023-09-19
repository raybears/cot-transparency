import pytest

from cot_transparency.formatters import name_to_stage1_formatter
from cot_transparency.formatters.interventions.valid_interventions import name_to_intervention
from tests.prompt_formatting.test_prompt_formatter import EMPIRE_OF_PANTS_EXAMPLE


@pytest.mark.parametrize(
    "intervention_name, formatter1, formatter2",
    [
        (
            "MixedFormatFewShotLabelOnly10",
            "NoCotPromptSenFormatter_NUMBERS_FULL_ANS_CHOICES_PAREN",
            "NoCotPromptSenFormatter_FOO_FULL_ANS_CHOICES_DOT",
        ),
        (
            "MixedFormatFewShot10",
            "CotPromptSenFormatter_NUMBERS_FULL_ANS_CHOICES_PAREN",
            "CotPromptSenFormatter_FOO_FULL_ANS_CHOICES_DOT",
        ),
    ],
)
def test_all_answers_same_format(intervention_name: str, formatter1: str, formatter2: str):
    """Test that going through the intervention pipeline with two different formatters, only the
    last message (i.e. the actual question) should be different the few shot labels should be the same"""

    question = EMPIRE_OF_PANTS_EXAMPLE

    Intervention = name_to_intervention(intervention_name)

    msgs1 = Intervention.intervene(question, name_to_stage1_formatter(formatter1), model="gpt-4")
    msgs2 = Intervention.intervene(question, name_to_stage1_formatter(formatter2), model="gpt-4")

    for i, (m1, m2) in enumerate(list(zip(msgs1, msgs2))[:-1]):
        assert m1.content == m2.content, f"messages {i} are different"

    # and check that the last answers are different
    assert msgs1[-1].content != msgs2[-1].content


if __name__ == "__main__":
    pytest.main([__file__])
