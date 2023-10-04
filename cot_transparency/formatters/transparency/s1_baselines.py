from cot_transparency.data_models.example_base import DataExampleBase, MultipleChoiceAnswer
from cot_transparency.data_models.messages import MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.instructions import (
    COT_ASSISTANT_PROMPT,
)
from cot_transparency.data_models.messages import ChatMessage


from typing import Optional
from cot_transparency.formatters.transparency.util import (
    SINGLE_MOST_LIKELY_ANSWER_COMPLETION,
    strip_given_all_of_the_above,
    reject_if_stop_tokens,
)
from cot_transparency.formatters.transparency.util import GIVEN_ALL_OF_THE_ABOVE

from cot_transparency.formatters.util import get_few_shot_prompts
from cot_transparency.model_apis import ModelType


class FormattersForTransparency(StageOneFormatter):
    has_none_of_the_above = True
    pass


class ZeroShotCOTUnbiasedTameraTFormatter(FormattersForTransparency):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        assert model is not None
        model_type = ModelType.from_model_name(model)
        msg = question.get_parsed_input_with_none_of_the_above()
        match model_type:
            case ModelType.completion | ModelType.chat_with_append_assistant:
                output = [
                    ChatMessage(role=MessageRole.user, content=msg),
                    ChatMessage(role=MessageRole.assistant, content=COT_ASSISTANT_PROMPT),
                ]
            case ModelType.chat:
                msg = msg + "\n\n" + "Please think step by step."
                output = [
                    ChatMessage(role=MessageRole.user, content=msg),
                ]

        return output

    @staticmethod
    def parse_answer(
        response: str, question: Optional[DataExampleBase] = None, model: Optional[str] = None
    ) -> Optional[str]:
        return "Extraction not implemented for this formatter as expected to run stage_two"


class FewShotCOTUnbiasedCompletionNoRoleTameraTFormatter(FormattersForTransparency):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        few_shots: list[tuple[ChatMessage, ChatMessage, MultipleChoiceAnswer]] = get_few_shot_prompts(
            question.get_parsed_input()
        )

        # we want to add two more messages
        few_shot_examples: list[ChatMessage] = []
        for q, ans, letter in few_shots:
            few_shot_examples.append(q.remove_role().add_question_prefix())
            few_shot_examples.append(ans.remove_role().add_answer_prefix())

            # Few shots all have Therefore, the answer is (X). So answer is always the last 3 characters
            single_best_answer = ChatMessage(
                role=MessageRole.none, content=f"{SINGLE_MOST_LIKELY_ANSWER_COMPLETION}{letter})."
            )
            few_shot_examples.append(single_best_answer)

        output = [
            ChatMessage(role=MessageRole.none, content=question.get_parsed_input()).add_question_prefix(),
            ChatMessage(role=MessageRole.none, content=COT_ASSISTANT_PROMPT).add_answer_prefix(),
        ]
        return few_shot_examples + output

    @staticmethod
    def parse_answer(
        response: str, question: Optional[DataExampleBase] = None, model: Optional[str] = None
    ) -> Optional[str]:
        ans = FewShotCOTUnbiasedTameraTFormatter.parse_answer(response)
        return strip_given_all_of_the_above(ans)


class FewShotCOTUnbiasedTameraTFormatter(FormattersForTransparency):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        few_shots: list[tuple[ChatMessage, ChatMessage, MultipleChoiceAnswer]] = get_few_shot_prompts(
            question.get_parsed_input()
        )

        # we want to add two more messages
        few_shot_examples: list[ChatMessage] = []
        for q, ans, letter in few_shots:
            few_shot_examples.append(q)
            few_shot_examples.append(ans)

            given_all_of_the_above = ChatMessage(role=MessageRole.user, content=GIVEN_ALL_OF_THE_ABOVE)
            few_shot_examples.append(given_all_of_the_above)

            single_best_answer = ChatMessage(
                role=MessageRole.assistant, content=f"The single, most likely answer is: ({letter})."
            )
            few_shot_examples.append(single_best_answer)

        output = [
            ChatMessage(role=MessageRole.user, content=question.get_parsed_input()),
            ChatMessage(role=MessageRole.assistant, content=COT_ASSISTANT_PROMPT),
        ]
        return few_shot_examples + output

    @staticmethod
    def parse_answer(
        response: str, question: Optional[DataExampleBase] = None, model: Optional[str] = None
    ) -> Optional[str]:
        return reject_if_stop_tokens(response)
