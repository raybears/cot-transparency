import json
from pathlib import Path

from pydantic import BaseModel

class MilesBBHRawData(BaseModel):
    # Already formatted to have the answer of A all the time
    idx: int
    inputs: str
    targets: list[str]
    multiple_choice_targets: list[str]
    multiple_choice_scores: list[int]
    split: str
    random_ans_idx: int
    parsed_inputs: str

class MilesBBHRawDataFolder(BaseModel):
    data: list[MilesBBHRawData]


BBH_TASK_LIST = [
    # "sports_understanding",
    # "snarks",
    # 'disambiguation_qa',
    # 'movie_recommendation',
    # 'causal_judgment',
    # 'date_understanding',
    # 'tracking_shuffled_objects_three_objects',
    # 'temporal_sequences',
    "ruin_names",
    # 'web_of_lies',
    # 'navigate',
    # 'logical_deduction_five_objects',
    # 'hyperbaton',
]


def format_initial_prompt(question: str) -> str:
    return f"""{question}"""


if __name__ == "__main__":
    # bbh is in data/bbh/task_name
    # read in the json file
    task_name = "ruin_names"
    # data/bbh/{task_name}/val_data.json
    json_path: Path = Path(f"data/bbh/{task_name}/val_data.json")
    with open(json_path, "r") as f:
        raw_data = json.load(f)
        # parse it into MilesBBHRawDataFolder
        data = MilesBBHRawDataFolder(**raw_data)
        print(data)
