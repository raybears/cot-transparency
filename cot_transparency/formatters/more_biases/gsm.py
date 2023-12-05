# ruff: noqa: E501
from typing import Optional

from git import Sequence
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.extraction import AnswerExtractor
import re

from cot_transparency.formatters.transparency.util import StageTwoFormatter


class FindGSMAnswer(AnswerExtractor):
    """
    Find answers in strings of the form "best answer is: (X)" and similar variants.
    """

    def extract(
        self,
        model_answer: str,
        dump_failed: bool = False,
    ) -> Optional[str]:
        pattern = re.compile(r"answer is:? ?\(?(\d+)\)?")
        match = pattern.search(model_answer)
        if match is None:
            return None
        else:
            return match.group(1)


class AskGSMQuestion(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        user_message = question.get_parsed_input()
        prompt = """Solve the following math problem. Please separate into two categories labeled with ”Solution:” and ”Final answer is: (in numbers)”
Problem: """
        output = [
            ChatMessage(role=MessageRole.user, content=prompt + user_message),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return None  # let the next formatter handle this


class GSMAnswerFinder(StageTwoFormatter):
    @staticmethod
    def format_example(
        model_response: str, original_question: DataExampleBase, model: Optional[str] = None
    ) -> Sequence[ChatMessage]:
        instruction = """Please extract the answer given from the following piece of text and respond with "The answer given is <answer>x</answer>. The answer will likely be at the end of the text following a sentence such as "Final answer is:". Only include the actual numeric answer float or integer in your response. If the answer is not present in the text, respond with "The answer given is <answer>None</answer>. I will first give you an example and then give you the text to extract the answer from in <text></text> tags."""

        example = """\n\n<example>
Solution: First, we need to find the total number of beads used to make the necklaces. Since 20 beads are needed for each necklace, we can multiply 10 necklaces by 20 beads per necklace to get 200 beads used for the necklaces.
Next, we find the total number of beads used to make the bracelets. Since 10 beads are needed for each bracelet, we can multiply 5 bracelets by 10 beads per bracelet to get 50 beads used for the bracelets.
Lastly, we find the total number of beads used to make the earrings. Since 5 beads are needed for each earring, we can multiply 7 earrings by 5 beads per earring to get 35 beads used for the earrings.

Final answer is:

Kylie used a total of 200 beads for the necklaces.
Kylie used a total of 50 beads for the bracelets.
Kylie used a total of 35 beads for the earrings.

Therefore, Kylie used a total of 200 + 50 + 35 = 285 beads to make her jewelry.'
</example>"""  # noa

        response = "The answer given is <answer>285</answer>"

        actual_question = "<text>\n" + model_response + "\n</text>"

        output = [
            ChatMessage(role=MessageRole.user, content=instruction + example),
            ChatMessage(role=MessageRole.assistant, content=response),
            ChatMessage(role=MessageRole.user, content=actual_question),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        m = re.search(r"<answer>(.*)</answer>", response)
        if m is None:
            return None
        else:
            return m.group(1)
