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
    ROMAN = "numerals"
    FOO = "foo"

    @property
    def answers_list(self) -> list[str]:
        choices_map = {
            ChoiceVariant.LETTERS: list(ascii_uppercase),
            ChoiceVariant.NUMBERS: [str(i) for i in range(1, 15)],
            ChoiceVariant.ROMAN: ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII"],
            ChoiceVariant.FOO: [
                "foo",
                "bar",
                "baz",
                "qux",
                "quux",
                "corge",
                "grault",
                "garply",
                "waldo",
                "fred",
                "plugh",
                "xyzzy",
                "thud",
            ],
        }
        return choices_map[self]


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


class IndicatorSeparator(str, Enum):
    DOT = "dot"
    PAREN = "paren"


class DataFormatSpec(BaseModel):
    choice_variant: ChoiceVariant = ChoiceVariant.LETTERS
    question_variant: QuestionPrefix = QuestionPrefix.NONE
    join_variant: JoinStr = JoinStr.ANS_CHOICES
    indicator_separator: IndicatorSeparator = IndicatorSeparator.PAREN

    def __str__(self):
        return (
            f"{self.choice_variant.name}_{self.question_variant.name}_"
            f"{self.join_variant.name}_{self.indicator_separator.name}"
        )

    @classmethod
    def init_random(cls, seed: int):
        rng = random.Random(seed)
        choice_variant = rng.choice(list(ChoiceVariant))
        question_variant = rng.choice(list(QuestionPrefix))
        join_variant = rng.choice(list(JoinStr))
        indicator_separator = rng.choice(list(IndicatorSeparator))
        return DataFormatSpec(
            choice_variant=choice_variant,
            question_variant=question_variant,
            join_variant=join_variant,
            indicator_separator=indicator_separator,
        )


def raise_if_not_multiple_choice_answer(string: str) -> MultipleChoiceAnswer:
    assert string in VALID_ANSWERS
    return string


def combine_indicator_with_separator(indicator: str, separator: IndicatorSeparator) -> str:
    match separator:
        case IndicatorSeparator.DOT:
            return f"{indicator}. "
        case IndicatorSeparator.PAREN:
            return f"({indicator}) "


class LetterAndOption(BaseModel):
    letter: MultipleChoiceAnswer
    option: str


class DataExampleBase(BaseModel, ABC):
    """We don't define the fields here because we want to be able to use this for any dataset but we define the api"""

    data_format: DataFormatSpec = DataFormatSpec()

    def to_variant(
        self,
        data_format_spec: DataFormatSpec,
    ) -> Self:
        c = self.copy()
        c.data_format = data_format_spec

        return c

    @property
    @abstractmethod
    def ground_truth(self) -> MultipleChoiceAnswer:
        """Please implement this method to return the ground truth answer"""
        raise NotImplementedError

    @property
    def ground_truth_indicator(self) -> str:
        return self.data_format.choice_variant.answers_list[self.ground_truth_idx()]

    @abstractmethod
    def _get_options(self) -> list[str]:
        """Please implement this method to return a list of options, without any letters"""
        raise NotImplementedError

    def get_options(self, include_none_of_the_above: bool = False) -> list[str]:
        options = self._get_options()
        if include_none_of_the_above:
            if "none" not in " ".join(options).lower():
                options.append("None of the above")
        return options

    @abstractmethod
    def _get_question(self) -> str:
        """Please implement this method to return the question, without any options"""
        raise NotImplementedError

    def ground_truth_idx(self) -> int:
        return ascii_uppercase.index(self.ground_truth)

    def _get_options_with_indicator(self, options: list[str]) -> str:
        output = []
        for idx, option in enumerate(options):
            choice_variant = self.data_format.choice_variant
            indicator = choice_variant.answers_list[idx]
            combined = combine_indicator_with_separator(indicator, self.data_format.indicator_separator)
            output.append(f"{combined}{option}")
        return "\n".join(output)

    @staticmethod
    def format_options_with_letters(options: list[LetterAndOption]) -> str:
        return "\n".join([f"({option.letter}) {option.option}" for option in options])

    @staticmethod
    def _get_lettered_options(options: list[str]) -> list[LetterAndOption]:
        return [
            LetterAndOption(letter=ascii_uppercase[idx], option=option)  # type: ignore
            for idx, option in enumerate(options)
        ]

    @property
    def biased_ans(self) -> MultipleChoiceAnswer:
        rng = random.Random(self.get_parsed_input())  # seed with question
        n_choices = len(self._get_options())
        biased_ans_idx = rng.randrange(0, n_choices)  # select random answer for bias metrics
        biased_ans_letter: MultipleChoiceAnswer = ascii_uppercase[biased_ans_idx]  # type: ignore
        return biased_ans_letter

    def hash(self) -> str:
        return deterministic_hash(self.get_parsed_input())

    def get_parsed_input_with_none_of_the_above(self) -> str:
        return self.get_parsed_input(include_none_of_the_above=True)

    # prompt sensitivity methods
    def get_parsed_input(
        self,
        include_none_of_the_above: bool = False,
    ) -> str:
        question = self._get_question()
        # check question doesn't start with question or q
        assert not question.lower().startswith("question") or question.lower().startswith("q")

        match self.data_format.question_variant:
            case QuestionPrefix.FULL:
                question = f"Question: {question}"
            case QuestionPrefix.SHORT:
                question = f"Q: {question}"
            case QuestionPrefix.NONE:
                pass

        choices = self._get_options(include_none_of_the_above=include_none_of_the_above)
        choices_str = self._get_options_with_indicator(choices)

        return f"{question}{self.data_format.join_variant.value}{choices_str}"


GenericDataExample = TypeVar("GenericDataExample", bound="DataExampleBase")
