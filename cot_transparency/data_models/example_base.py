from abc import ABC, abstractmethod
from enum import Enum
import random
from typing import Literal, TypeVar
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

    letters = "letters"
    numbers = "numbers"
    numerals = "numerals"
    foo = "foo"


class QuestionPrefix(str, Enum):
    question_prefix = "Question:"
    q_prefix = "Q:"
    no_prefix = None


class JoinStr(str, Enum):
    ans_choices = "\n\nAnswer choices:\n"
    options = "\n\nOptions:\n"


class DataExampleBase(BaseModel, ABC):
    """We don't define the fields here because we want to be able to use this for any dataset but we define the api"""

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

    def _get_options_with_indicator(
        self, options: list[str], choice_variant: ChoiceVariant = ChoiceVariant.letters
    ) -> str:
        output = []
        for idx, option in enumerate(options):
            match choice_variant:
                case ChoiceVariant.letters:
                    indicator = ascii_uppercase[idx]
                case ChoiceVariant.numbers:
                    indicator = idx + 1
                case ChoiceVariant.numerals:
                    indicator = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"][idx]
                case ChoiceVariant.foo:
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
        choice_variant: ChoiceVariant = ChoiceVariant.letters,
        question_variant: QuestionPrefix = QuestionPrefix.no_prefix,
        join_variant: JoinStr = JoinStr.ans_choices,
    ) -> str:
        question = self._get_question()
        # check question doesn't start with question or q
        assert not question.lower().startswith("question") or question.lower().startswith("q")

        match question_variant:
            case QuestionPrefix.question_prefix:
                question = f"Question: {question}"
            case QuestionPrefix.q_prefix:
                question = f"Q: {question}"
            case QuestionPrefix.no_prefix:
                pass

        choices = self._get_options()
        choices_str = self._get_options_with_indicator(choices, choice_variant=choice_variant)

        return f"{question}{join_variant.value}{choices_str}"


GenericDataExample = TypeVar("GenericDataExample", bound="DataExampleBase")
