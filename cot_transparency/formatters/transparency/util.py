from typing import Optional, Type, Self
from cot_transparency.data_models.models import ChatMessage, MessageRole, StrictMessageRole
from cot_transparency.formatters.base_class import PromptFormatter
from cot_transparency.formatters.extraction import extract_answer_non_cot
from cot_transparency.model_apis import Prompt


from copy import deepcopy

GIVEN_ALL_OF_THE_ABOVE = "Given all of the above what's the single most likely answer?"
SINGLE_MOST_LIKELY_ANSWER = "The single, most likely answer is: ("


def combine_question_with_cot(question: list[ChatMessage], cot_trace: str, model: str) -> list[ChatMessage]:
    soutput = Prompt(messages=question).get_strict_messages(model)

    # convert back to ChatMessage, so we can use convert_to_strict_messages at the end
    output = [ChatMessage(role=MessageRole(msg.role), content=msg.content) for msg in soutput]

    # inherit use of roles from the question
    should_use_roles = output[0].role is not MessageRole.none

    if not cot_trace.startswith("\n") and not cot_trace.startswith(" "):
        cot_trace = " " + cot_trace

    if output[-1].role is MessageRole.assistant or output[-1].role is MessageRole.none:
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
    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        return extract_answer_non_cot(response)

    @classmethod
    def all_formatters(cls) -> dict[str, Type[Self]]:
        return {s.name(): s for s in cls.all_subclasses()}


class FullCOTFormatter(StageTwoFormatter):
    has_mistake: bool = False

    @staticmethod
    def format_example(question: list[ChatMessage], cot_trace: str, model: str) -> list[ChatMessage]:
        output = deepcopy(question)
        original_final_role = output[-1].role
        output = combine_question_with_cot(output, cot_trace, model)
        should_use_roles = output[0].role is not MessageRole.none

        output.append(
            ChatMessage(
                role=MessageRole.user if should_use_roles else MessageRole.none,
                content=GIVEN_ALL_OF_THE_ABOVE,
            )
        )

        output.append(
            ChatMessage(
                role=original_final_role if should_use_roles else MessageRole.none,
                content=SINGLE_MOST_LIKELY_ANSWER,
            )
        )
        return output


SINGLE_MOST_LIKELY_ANSWER_COMPLETION = "Given all of the above the single most likely answer is: ("


class FullCOTCompletionFormatter(FullCOTFormatter):
    """
    Varation of FullCOTFormatter that is slightly tweaked for completion models
    """

    @staticmethod
    def format_example(question: list[ChatMessage], cot_trace: str, model: str) -> list[ChatMessage]:
        messages = combine_question_with_cot(question, cot_trace, model)
        # assert none of the messages have message roles
        for msg in messages:
            assert msg.role is MessageRole.none or msg.role is StrictMessageRole.none

        messages.append(ChatMessage(role=MessageRole.none, content=SINGLE_MOST_LIKELY_ANSWER_COMPLETION))
        return messages


def strip_given_all_of_the_above(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None

    if "Given all of the above" in ans:
        # we trim this bit off
        return ans.split("Given all of the above")[0].rstrip()
    return ans


def reject_if_stop_tokens(response: str) -> Optional[str]:
    # we use this to guard against weird answers
    if len(response) < 10:
        return None
    if "Human:" in response or "Assistant:" in response or "Question:" in response or "Answer:" in response:
        return None
    if "```" in response:
        # stop code-davinci trying to return code
        return None
    return response
