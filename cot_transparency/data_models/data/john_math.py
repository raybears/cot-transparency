from pathlib import Path
from string import ascii_uppercase

import pandas as pd
from slist import Slist

from cot_transparency.data_models.example_base import (
    DataExampleBase,
    IndicatorAndOption,
    MultipleChoiceAnswer,
    raise_if_not_multiple_choice_answer,
)


class JohnMath(DataExampleBase):
    # John's hard math problems
    question: str
    level: float
    type: str
    solution: str
    correct_answer: str
    negative_answer: str
    negative_solution: str
    biased_answer: MultipleChoiceAnswer

    @property
    def biased_ans(self) -> MultipleChoiceAnswer:
        # override the biased answer to be the one that is in the dataset
        return self.biased_answer

    def deterministic_randomized_options(self) -> Slist[IndicatorAndOption]:
        answers = Slist([self.correct_answer, self.negative_answer])
        # 50% chance for correct answer to be A
        answer_list: Slist[IndicatorAndOption] = (
            Slist(["A", "B"])
            .shuffle(seed=self.question)
            .zip(answers)
            .map(
                lambda tup: IndicatorAndOption(
                    indicator=raise_if_not_multiple_choice_answer(tup[0]),
                    option=tup[1],
                )
            )
        ).shuffle(seed=self.question)
        return answer_list

    @property
    def ground_truth(self) -> MultipleChoiceAnswer:
        found_answer: int = (
            self.deterministic_randomized_options().find_one_idx_or_raise(
                lambda x: x.option == self.correct_answer
            )
        )
        return ascii_uppercase[found_answer]  # type: ignore

    def _get_options(self) -> list[str]:
        return self.deterministic_randomized_options().map(lambda x: x.option)

    def _get_question(self) -> str:
        return self.question


def get_john_math(path: Path) -> Slist[JohnMath]:
    # this big brain code makes two examples for each row.
    # One with A as the biased answer, and one with B as the biased answer
    df = pd.read_csv(path)
    biased_a = {"biased_answer": "A"}
    biased_b = {"biased_answer": "B"}
    return Slist(
        [JohnMath(**dict(**row) | biased_a), JohnMath(**dict(**row) | biased_b)]
        for _, row in df.iterrows()
    ).flatten_list()


def get_john_math_level_1() -> Slist[JohnMath]:
    return get_john_math(
        Path("./data/john_math/train_balanced_level1.csv"),
    )


def get_john_math_level_2() -> Slist[JohnMath]:
    return get_john_math(
        Path("./data/john_math/train_balanced_level2.csv"),
    )


def get_john_math_level_3() -> Slist[JohnMath]:
    return get_john_math(
        Path("./data/john_math/train_balanced_level3.csv"),
    )


def get_john_math_level_4() -> Slist[JohnMath]:
    return get_john_math(
        Path("./data/john_math/train_balanced_level4.csv"),
    )


def get_john_math_level_5() -> Slist[JohnMath]:
    return get_john_math(
        Path("./data/john_math/train_balanced_level5.csv"),
    )
