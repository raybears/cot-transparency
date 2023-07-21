from cot_transparency.model_apis import format_for_anthropic, format_for_openai_chat
from cot_transparency.data_models.models_v2 import ChatMessages
from cot_transparency.formatters import ZeroShotSycophancyNoRoleFormatter
from tests.test_prompt_formatter import EMPIRE_OF_PANTS_EXAMPLE


from pytest import raises


def test_role_required_openai_chat():
    test_example = EMPIRE_OF_PANTS_EXAMPLE
    formatted_prompt: list[ChatMessages] = ZeroShotSycophancyNoRoleFormatter().format_example(test_example)
    with raises(ValueError):
        format_for_openai_chat(formatted_prompt)


def test_role_required_anthropic():
    test_example = EMPIRE_OF_PANTS_EXAMPLE
    formatted_prompt: list[ChatMessages] = ZeroShotSycophancyNoRoleFormatter().format_example(test_example)
    with raises(ValueError):
        format_for_anthropic(formatted_prompt)
