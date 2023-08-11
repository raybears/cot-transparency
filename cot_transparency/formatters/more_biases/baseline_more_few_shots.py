from abc import ABC, abstractmethod
from typing import Optional, Type, final

from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import ChatMessage, MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.consistency_prompting.few_shots import consistency_few_shot, few_shots_to_sample
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.extraction import extract_answer
from cot_transparency.formatters.instructions import add_verbalize_instruction_to_question, COT_ASSISTANT_PROMPT
from scripts.biased_wrong_ans import sample_few_shots_cot


class Intervention(ABC):
    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @abstractmethod
    def hook(self, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Define a hook that can be used to intervene in the formatting process."""
        pass

    @final  # Please don't override this unless you know what you are doing.
    def intervene(self, formatter: Type[StageOneFormatter]) -> Type[StageOneFormatter]:
        intervention_self = self

        class NewFormatter(formatter):
            @staticmethod
            def format_example(question: DataExampleBase) -> list[ChatMessage]:
                messages = formatter.format_example(question)
                new_messages = self.hook(question, messages)
                return new_messages

            @staticmethod
            def parse_answer(response: str) -> Optional[str]:
                return formatter.parse_answer(response)

            @classmethod
            def name(cls) -> str:
                return f"{intervention_self.name()}_{formatter.name()}"

        return NewFormatter


def prepend_to_front_first_user_message(messages: list[ChatMessage], prepend: str) -> list[ChatMessage]:
    """Prepend a string to the first user message."""
    new_messages = []
    for m in messages:
        if m.role == MessageRole.user:
            new_messages.append(ChatMessage(role=MessageRole.user, content=prepend + m.content))
        else:
            new_messages.append(m)
    return messages


class PairedConsistency3(Intervention):

    def hook(self, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=sample_few_shots_cot(
                few_shots_to_sample, seed=question.hash(), n=5
            ).convert_to_completion_str(),
        )
        return new


class PairedConsistency5(Intervention):

    def hook(self, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=sample_few_shots_cot(
                few_shots_to_sample, seed=question.hash(), n=5
            ).convert_to_completion_str(),
        )
        return new

class PairedConsistency10(Intervention):

    def hook(self, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=sample_few_shots_cot(
                few_shots_to_sample, seed=question.hash(), n=10
            ).convert_to_completion_str(),
        )
        return new


five_shot_intervention = PairedConsistency5()
PairedConsistency5_ZeroShotCOTSycophancyFormatter: Type[StageOneFormatter] = five_shot_intervention.intervene(
    ZeroShotCOTSycophancyFormatter
)
PairedConsistency5_ZeroShotCOTUnbiasedFormatter: Type[StageOneFormatter] = five_shot_intervention.intervene(
    ZeroShotCOTUnbiasedFormatter
)
