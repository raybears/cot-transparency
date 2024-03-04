from pathlib import Path
from string import ascii_uppercase
from typing import List


from cot_transparency.data_models.example_base import (
    DataExampleBase,
    MultipleChoiceAnswer,
)
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel


class DiscrimEval(DataExampleBase):
    filled_template: str
    age: float
    gender: str
    race: str

    def _get_options(self) -> list[str]:
        return ["Yes", "No"]

    def _get_question(self) -> str:
        return self.filled_template

    @property
    def _ground_truth(self) -> MultipleChoiceAnswer:
        return ascii_uppercase[0]  # type: ignore


def discrim_eval_baseline() -> List[DiscrimEval]:
    path = Path("./data/discrim-eval/data.jsonl")
    data = read_jsonl_file_into_basemodel(path, DiscrimEval)
    filtered_data = data.filter(lambda x: x.age == 60.0 and x.gender == "male" and x.race == "white")
    return filtered_data


def discrim_eval_black_fixed() -> List[DiscrimEval]:
    path = Path("./data/discrim-eval/data.jsonl")
    data = read_jsonl_file_into_basemodel(path, DiscrimEval)
    filtered_data = data.filter(lambda x: x.race == "Black" and x.age == 60.0 and x.gender == "male")
    return filtered_data


def discrim_eval_black() -> List[DiscrimEval]:
    path = Path("./data/discrim-eval/data.jsonl")
    data = read_jsonl_file_into_basemodel(path, DiscrimEval)
    filtered_data = data.filter(lambda x: x.race == "Black")
    return filtered_data


def discrim_eval_hispanic() -> List[DiscrimEval]:
    path = Path("./data/discrim-eval/data.jsonl")
    data = read_jsonl_file_into_basemodel(path, DiscrimEval)
    filtered_data = data.filter(lambda x: x.race == "Hispanic")
    return filtered_data


def discrim_eval_native_american() -> List[DiscrimEval]:
    path = Path("./data/discrim-eval/data.jsonl")
    data = read_jsonl_file_into_basemodel(path, DiscrimEval)
    filtered_data = data.filter(lambda x: x.race == "Native American")
    return filtered_data


def discrim_eval_asian() -> List[DiscrimEval]:
    path = Path("./data/discrim-eval/data.jsonl")
    data = read_jsonl_file_into_basemodel(path, DiscrimEval)
    filtered_data = data.filter(lambda x: x.race == "Asian")
    return filtered_data


def discrim_eval_female() -> List[DiscrimEval]:
    path = Path("./data/discrim-eval/data.jsonl")
    data = read_jsonl_file_into_basemodel(path, DiscrimEval)
    filtered_data = data.filter(lambda x: x.gender == "female")
    return filtered_data


def discrim_eval_non_binary() -> List[DiscrimEval]:
    path = Path("./data/discrim-eval/data.jsonl")
    data = read_jsonl_file_into_basemodel(path, DiscrimEval)
    filtered_data = data.filter(lambda x: x.gender == "non-binary")
    return filtered_data


def discrim_eval_age_20() -> List[DiscrimEval]:
    path = Path("./data/discrim-eval/data.jsonl")
    data = read_jsonl_file_into_basemodel(path, DiscrimEval)
    filtered_data = data.filter(lambda x: x.age == 20.0)
    return filtered_data


def discrim_eval_age_30() -> List[DiscrimEval]:
    path = Path("./data/discrim-eval/data.jsonl")
    data = read_jsonl_file_into_basemodel(path, DiscrimEval)
    filtered_data = data.filter(lambda x: x.age == 30.0)
    return filtered_data


def discrim_eval_age_40() -> List[DiscrimEval]:
    path = Path("./data/discrim-eval/data.jsonl")
    data = read_jsonl_file_into_basemodel(path, DiscrimEval)
    filtered_data = data.filter(lambda x: x.age == 40.0)
    return filtered_data


def discrim_eval_age_50() -> List[DiscrimEval]:
    path = Path("./data/discrim-eval/data.jsonl")
    data = read_jsonl_file_into_basemodel(path, DiscrimEval)
    filtered_data = data.filter(lambda x: x.age == 50.0)
    return filtered_data


def discrim_eval_age_60() -> List[DiscrimEval]:
    path = Path("./data/discrim-eval/data.jsonl")
    data = read_jsonl_file_into_basemodel(path, DiscrimEval)
    filtered_data = data.filter(lambda x: x.age == 60.0)
    return filtered_data


def discrim_eval_age_70() -> List[DiscrimEval]:
    path = Path("./data/discrim-eval/data.jsonl")
    data = read_jsonl_file_into_basemodel(path, DiscrimEval)
    filtered_data = data.filter(lambda x: x.age == 70.0)
    return filtered_data


def discrim_eval_age_80() -> List[DiscrimEval]:
    path = Path("./data/discrim-eval/data.jsonl")
    data = read_jsonl_file_into_basemodel(path, DiscrimEval)
    filtered_data = data.filter(lambda x: x.age == 80.0)
    return filtered_data


def discrim_eval_age_90() -> List[DiscrimEval]:
    path = Path("./data/discrim-eval/data.jsonl")
    data = read_jsonl_file_into_basemodel(path, DiscrimEval)
    filtered_data = data.filter(lambda x: x.age == 90.0)
    return filtered_data


def discrim_eval_age_100() -> List[DiscrimEval]:
    path = Path("./data/discrim-eval/data.jsonl")
    data = read_jsonl_file_into_basemodel(path, DiscrimEval)
    filtered_data = data.filter(lambda x: x.age == 100.0)
    return filtered_data


DISCRIM_EVAL_TASKS_LIST = [
    "discrim_eval_baseline",
    "discrim_eval_black_fixed",
    "discrim_eval_black",
    "discrim_eval_hispanic",
    "discrim_eval_native_american",
    "discrim_eval_asian",
    "discrim_eval_female",
    "discrim_eval_non_binary",
    "discrim_eval_age_20",
    "discrim_eval_age_30",
    "discrim_eval_age_40",
    "discrim_eval_age_50",
    "discrim_eval_age_60",
    "discrim_eval_age_70",
    "discrim_eval_age_80",
    "discrim_eval_age_90",
    "discrim_eval_age_100",
]
