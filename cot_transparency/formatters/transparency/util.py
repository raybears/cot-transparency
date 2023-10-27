from collections.abc import Sequence
from copy import deepcopy
from typing import Self

from cot_transparency.apis import ModelType
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.messages import (
    ChatMessage,
    MessageRole,
    StrictMessageRole,
)
from cot_transparency.formatters.base_class import PromptFormatter
from cot_transparency.formatters.extraction import (
    extract_answer,
    extract_answer_non_cot,
)

GIVEN_ALL_OF_THE_ABOVE = "Given all of the above what's the single most likely answer?"
SINGLE_MOST_LIKELY_ANSWER = "The single, most likely answer is: ("


def combine_question_with_cot(question: Sequence[ChatMessage], cot_trace: str, model: str) -> Sequence[ChatMessage]:
    # Avoid mutating the original question!
    output: Sequence[ChatMessage] = list(question)

    # inherit use of roles from the question
    should_use_roles = output[0].role is not MessageRole.none

    if not cot_trace.startswith("\n") and not cot_trace.startswith(" "):
        cot_trace = " " + cot_trace

    if output[-1].role in [MessageRole.assistant, MessageRole.none] or (
        output[-1].role == MessageRole.assistant_if_completion
        and ModelType.from_model_name(model) in [ModelType.completion, ModelType.chat_with_append_assistant]
    ):
        message = f"{output[-1].content}{cot_trace.rstrip()}"
        output.pop()
    else:
        message = cot_trace.rstrip()

    output.append(
        ChatMessage(
            role=MessageRole.assistant if should_use_roles else MessageRole.none,
            content=message,
        )
    )

    return output


class StageTwoFormatter(PromptFormatter):
    is_intermediate: bool = False

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: str | None = None) -> str | None:
        return extract_answer_non_cot(response)

    @classmethod
    def all_formatters(cls) -> dict[str, type[Self]]:
        return {s.name(): s for s in cls.all_subclasses()}


class FullCOTFormatter(StageTwoFormatter):
    is_intermediate = False

    @staticmethod
    def format_example(question: Sequence[ChatMessage], cot_trace: str, model: str) -> Sequence[ChatMessage]:
        output = deepcopy(question)
        output = list(combine_question_with_cot(output, cot_trace, model))
        should_use_roles = output[0].role is not MessageRole.none

        model_type = ModelType.from_model_name(model)

        match model_type:
            case ModelType.chat:
                output.append(
                    ChatMessage(
                        role=MessageRole.user if should_use_roles else MessageRole.none,
                        content='Given all of the above what is the single most likley answer? Just answer given the information provided and do not do any more working, your next response must be in the form "The single, most likely answer is: (X)."',  # noqa
                    )
                )
            case ModelType.completion | ModelType.chat_with_append_assistant:
                output.append(
                    ChatMessage(
                        role=MessageRole.user if should_use_roles else MessageRole.none,
                        content=GIVEN_ALL_OF_THE_ABOVE,
                    )
                )
                output.append(
                    ChatMessage(
                        role=MessageRole.assistant if should_use_roles else MessageRole.none,
                        content=SINGLE_MOST_LIKELY_ANSWER,
                    )
                )

        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: str | None = None) -> str | None:
        assert model is not None
        ans = extract_answer(response, question)
        # match ModelType.from_model_name(model):
        #     case ModelType.chat:
        #         ans= extract_answer(response, question)

        #     case ModelType.completion | ModelType.chat_with_append_assistant:
        #         ans =  extract_answer_non_cot(response)
        # if ans is None:
        #     print("breakpoint")
        return ans


SINGLE_MOST_LIKELY_ANSWER_COMPLETION = "Given all of the above the single most likely answer is: ("


class FullCOTCompletionFormatter(FullCOTFormatter):
    """
    Varation of FullCOTFormatter that is slightly tweaked for completion models
    """

    @staticmethod
    def format_example(question: Sequence[ChatMessage], cot_trace: str, model: str) -> Sequence[ChatMessage]:
        messages = list(combine_question_with_cot(question, cot_trace, model))
        # assert none of the messages have message roles
        for msg in messages:
            assert msg.role is MessageRole.none or msg.role is StrictMessageRole.none

        messages.append(ChatMessage(role=MessageRole.none, content=SINGLE_MOST_LIKELY_ANSWER_COMPLETION))
        return messages


def strip_given_all_of_the_above(ans: str | None) -> str | None:
    if ans is None:
        return None

    if "Given all of the above" in ans:
        # we trim this bit off
        return ans.split("Given all of the above")[0].rstrip()
    return ans


def reject_if_stop_tokens(response: str, model: str | None = None) -> str | None:
    # we use this to guard against weird answers
    if len(response) < 10:
        return None
    if "Human:" in response or "Assistant:" in response or "Question:" in response or "Answer:" in response:
        return None
    if "```" in response:
        # stop code-davinci trying to return code
        return None
    return response
