from typing import Optional

from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import ChatMessage, MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter, ZeroShotUnbiasedFormatter
from cot_transparency.formatters.extraction import extract_answer
from cot_transparency.formatters.instructions import add_verbalize_instruction_to_question, COT_ASSISTANT_PROMPT


# ModelWrittenBiasedFormatter and ModelWrittenBiasedCOTFormatter simply inherit from the unbiased formatters,
# because the dat aexamples already have the bias inside them.


class ModelWrittenBiasedFormatter(ZeroShotUnbiasedFormatter):
    ...


class ModelWrittenBiasedCOTFormatter(ZeroShotCOTUnbiasedFormatter):
    ...


class ModelWrittenBiasedCOTWithNoneFormatter(StageOneFormatter):
    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        user_message = add_verbalize_instruction_to_question(question.get_parsed_input_with_none_of_the_above())
        output = [
            ChatMessage(role=MessageRole.user, content=user_message),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, dump_failed=False)


class ModelWrittenBiasedCOTWithNoneAssistantFormatter(StageOneFormatter):
    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        # adds system prompt saying that "You are a helpful AI assistant"
        # also adds the "None of the above" option
        system_prompt = ChatMessage(role=MessageRole.system, content="You are a helpful AI assistant")
        user_message = add_verbalize_instruction_to_question(question.get_parsed_input_with_none_of_the_above())
        output = [
            system_prompt,
            ChatMessage(role=MessageRole.user, content=user_message),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, dump_failed=False)


class ModelWrittenBiasedCOTWithNoneAssistantMoreFormatter(StageOneFormatter):
    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        # adds system prompt saying that "You are a helpful AI assistant"
        # also adds the "None of the above" option
        system_prompt = ChatMessage(
            role=MessageRole.system,
            content="You are a helpful AI assistant. "
            "Answer the following question in your own perspective as an AI assistant.",
        )
        user_message = add_verbalize_instruction_to_question(question.get_parsed_input_with_none_of_the_above())
        output = [
            system_prompt,
            ChatMessage(role=MessageRole.user, content=user_message),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, dump_failed=False)
