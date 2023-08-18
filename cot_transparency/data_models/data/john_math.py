from pathlib import Path
from string import ascii_uppercase

from slist import Slist

from cot_transparency.data_models.example_base import DataExampleBase, MultipleChoiceAnswer
from cot_transparency.json_utils.read_write import read_base_model_from_csv


class JohnMath(DataExampleBase):
    # John's hard math problems
    question: str
    level: float
    type: str
    solution: str
    correct_answer: str
    negative_answer: str
    negative_solution: str

    def deterministic_randomized_options(self) -> Slist[str]:
        # 50% chance for correct answer to be A
        answer_list: Slist[str] = Slist([self.correct_answer, self.negative_answer]).shuffle(seed=self.question)
        return answer_list

    @property
    def ground_truth(self) -> MultipleChoiceAnswer:
        found_answer: int = self.deterministic_randomized_options().find_one_idx_or_raise(
            lambda x: x == self.correct_answer
        )
        return ascii_uppercase[found_answer]  # type: ignore

    def _get_options(self) -> list[str]:
        return self.deterministic_randomized_options()

    def _get_question(self) -> str:
        return self.question


def get_john_math_level_3() -> Slist[JohnMath]:
    return read_base_model_from_csv(
        Path("./data/john_math/train_balanced_level3.csv"),
        JohnMath,
    )


def get_john_math_level_4() -> Slist[JohnMath]:
    return read_base_model_from_csv(
        Path("./data/john_math/train_balanced_level4.csv"),
        JohnMath,
    )


def get_john_math_level_5() -> Slist[JohnMath]:
    return read_base_model_from_csv(
        Path("./data/john_math/train_balanced_level5.csv"),
        JohnMath,
    )
