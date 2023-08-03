import random
from string import ascii_uppercase
from cot_transparency.data_models.models import MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.instructions import (
    COT_ASSISTANT_PROMPT,
)
from cot_transparency.data_models.data.bbh import DataExampleBase
from cot_transparency.data_models.models import ChatMessage


from typing import Optional
from cot_transparency.formatters.transparency.util import (
    SINGLE_MOST_LIKELY_ANSWER_COMPLETION,
)
from cot_transparency.formatters.transparency.util import GIVEN_ALL_OF_THE_ABOVE

from cot_transparency.formatters.util import load_few_shots


class FormattersForTransparency(StageOneFormatter):
    pass


class ZeroShotCOTUnbiasedTameraTFormatter(FormattersForTransparency):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase) -> list[ChatMessage]:
        output = [
            ChatMessage(role=MessageRole.user, content=question.get_parsed_input()),
            ChatMessage(role=MessageRole.assistant, content=COT_ASSISTANT_PROMPT),
        ]
        return output

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        return "Extraction not implemented for this formatter as expected to run stage_two"


class FewShotCOTUnbiasedCompletionNoRoleTameraTFormatter(FormattersForTransparency):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase) -> list[ChatMessage]:
        few_shots: list[tuple[ChatMessage, ChatMessage]] = load_few_shots("./data/ethan_few_shot.txt")
        more_few_shots: list[tuple[ChatMessage, ChatMessage]] = load_few_shots("./data/gpt4_generated_few_shot.txt")

        few_shots = few_shots + more_few_shots
        random.Random(question.get_parsed_input()).shuffle(few_shots)

        # we want to add two more messages
        few_shot_examples: list[ChatMessage] = []
        for msg in few_shots:
            few_shot_examples.append(msg[0].remove_role().add_question_prefix())
            few_shot_examples.append(msg[1].remove_role().add_answer_prefix())

            # Few shots all have Therefore, the answer is (X). So answer is always the last 3 characters
            answer_from_few_shot = msg[1].content[-3]
            assert answer_from_few_shot in ascii_uppercase
            single_best_answer = ChatMessage(
                role=MessageRole.none, content=f"{SINGLE_MOST_LIKELY_ANSWER_COMPLETION}{answer_from_few_shot})."
            )
            few_shot_examples.append(single_best_answer)

        output = [
            ChatMessage(role=MessageRole.none, content=question.get_parsed_input()).add_question_prefix(),
            ChatMessage(role=MessageRole.none, content=COT_ASSISTANT_PROMPT).add_answer_prefix(),
        ]
        return few_shot_examples + output

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        ans = FewShotCOTUnbiasedTameraTFormatter.parse_answer(response)
        if ans is None:
            return None

        if "Given all of the above" in ans:
            # we trim this bit off
            return ans.split("Given all of the above")[0].rstrip()
        return ans


class FewShotCOTUnbiasedTameraTFormatter(FormattersForTransparency):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase) -> list[ChatMessage]:
        few_shots: list[tuple[ChatMessage, ChatMessage]] = load_few_shots("./data/ethan_few_shot.txt")
        more_few_shots: list[tuple[ChatMessage, ChatMessage]] = load_few_shots("./data/gpt4_generated_few_shot.txt")

        few_shots = few_shots + more_few_shots
        random.Random(question.get_parsed_input()).shuffle(few_shots)

        # we want to add two more messages
        few_shot_examples: list[ChatMessage] = []
        for msg in few_shots:
            few_shot_examples.append(msg[0])
            few_shot_examples.append(msg[1])

            given_all_of_the_above = ChatMessage(role=MessageRole.user, content=GIVEN_ALL_OF_THE_ABOVE)
            few_shot_examples.append(given_all_of_the_above)

            # Few shots all have Therefore, the answer is (X). So answer is always the last 3 characters
            answer_from_few_shot = msg[1].content[-3]
            assert answer_from_few_shot in ascii_uppercase
            single_best_answer = ChatMessage(
                role=MessageRole.assistant, content=f"The single, most likely answer is: ({answer_from_few_shot})."
            )
            few_shot_examples.append(single_best_answer)

        output = [
            ChatMessage(role=MessageRole.user, content=question.get_parsed_input()),
            ChatMessage(role=MessageRole.assistant, content=COT_ASSISTANT_PROMPT),
        ]
        return few_shot_examples + output

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        # we use this to guard against weird answers
        if len(response) < 10:
            return None
        if "Human:" in response or "Assistant:" in response or "Question:" in response or "Answer:" in response:
            return None
        if "```" in response:
            # stop code-davinci trying to return code
            return None

        return response
