from abc import ABC, abstractmethod
from enum import Enum
import random
from typing import Literal, Self, TypeVar
from pydantic import BaseModel
from string import ascii_uppercase

from cot_transparency.util import deterministic_hash

MultipleChoiceAnswer = Literal[
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
]
VALID_ANSWERS: set[MultipleChoiceAnswer] = {
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
}


class ChoiceVariant(str, Enum):
    """Prompt variants for prompt sensitivity analysis"""

    LETTERS = "letters"
    NUMBERS = "numbers"
    NUMERALS = "numerals"
    FOO = "foo"


class QuestionPrefix(Enum):
    FULL = "Question:"
    SHORT = "Q:"
    NONE = None

    def __str__(self):
        if self.value is not None:
            return self.value
        return ""


class JoinStr(str, Enum):
    ANS_CHOICES = "\n\nAnswer choices:\n"
    OPTIONS = "\n\nOptions:\n"


class DataExampleBase(BaseModel, ABC):
    """We don't define the fields here because we want to be able to use this for any dataset but we define the api"""

    # default question format
    choice_variant: ChoiceVariant = ChoiceVariant.LETTERS
    question_variant: QuestionPrefix = QuestionPrefix.NONE
    join_variant: JoinStr = JoinStr.ANS_CHOICES

    def to_variants(
        self,
        choice_variant: ChoiceVariant = ChoiceVariant.LETTERS,
        question_variant: QuestionPrefix = QuestionPrefix.NONE,
        join_variant: JoinStr = JoinStr.ANS_CHOICES,
    ) -> Self:
        c = self.copy()
        c.choice_variant = choice_variant
        c.question_variant = question_variant
        c.join_variant = join_variant
        return c

    @property
    @abstractmethod
    def ground_truth(self) -> MultipleChoiceAnswer:
        raise NotImplementedError

    @abstractmethod
    def _get_options(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def _get_question(self) -> str:
        raise NotImplementedError

    def ground_truth_idx(self) -> int:
        return ascii_uppercase.index(self.ground_truth)

    def _get_options_with_indicator(self, options: list[str]) -> str:
        output = []
        for idx, option in enumerate(options):
            match self.choice_variant:
                case ChoiceVariant.LETTERS:
                    indicator = ascii_uppercase[idx]
                case ChoiceVariant.NUMBERS:
                    indicator = idx + 1
                case ChoiceVariant.NUMERALS:
                    indicator = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"][idx]
                case ChoiceVariant.FOO:
                    indicator = ["foo", "bar", "baz", "qux", "quux", "corge", "grault", "garply", "waldo"][idx]
            output.append(f"({indicator}) {option}")
        return "\n".join(output)

    def get_parsed_input_with_none_of_the_above(self) -> str:
        question = self._get_question()
        options = self._get_options()
        if "none" not in " ".join(options).lower():
            options.append("None of the above")
        options_with_letters = self._get_options_with_indicator(options)
        return f"{question}\n\nAnswer choices:\n{options_with_letters}"

    @property
    def biased_ans(self) -> MultipleChoiceAnswer:
        rng = random.Random(self.get_parsed_input())  # seed with question
        n_choices = len(self._get_options())
        biased_ans_idx = rng.randrange(0, n_choices)  # select random answer for bias metrics
        biased_ans_letter: MultipleChoiceAnswer = ascii_uppercase[biased_ans_idx]  # type: ignore
        return biased_ans_letter

    def hash(self) -> str:
        return deterministic_hash(self.get_parsed_input())

    # prompt sensitivity methods
    def get_parsed_input(
        self,
    ) -> str:
        question = self._get_question()
        # check question doesn't start with question or q
        assert not question.lower().startswith("question") or question.lower().startswith("q")

        match self.question_variant:
            case QuestionPrefix.FULL:
                question = f"Question: {question}"
            case QuestionPrefix.SHORT:
                question = f"Q: {question}"
            case QuestionPrefix.NONE:
                pass

        choices = self._get_options()
        choices_str = self._get_options_with_indicator(choices)

        return f"{question}{self.join_variant.value}{choices_str}"


GenericDataExample = TypeVar("GenericDataExample", bound="DataExampleBase")
