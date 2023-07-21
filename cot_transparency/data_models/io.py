from typing import Union
from cot_transparency.data_models.models_v2 import ExperimentJsonFormat, StageTwoExperimentJsonFormat


import json
from glob import glob
from pathlib import Path

from cot_transparency.util import safe_file_write

LoadedJsonType = Union[dict[Path, ExperimentJsonFormat], dict[Path, StageTwoExperimentJsonFormat]]


class ExpLoader:
    @staticmethod
    def get_paths(exp_dir: str) -> list[str]:
        paths = glob(f"{exp_dir}/*/*/*.json", recursive=True)
        if paths:
            print(f"Found {len(paths)} json files")
        else:
            raise FileNotFoundError(f"Could not find any json files in {exp_dir}")
        return paths

    @staticmethod
    def get_stage(exp_dir: str) -> int:
        paths = ExpLoader.get_paths(exp_dir)
        first_exp = json.load(open(paths[0]))
        return int(first_exp["stage"])

    @staticmethod
    def stage_two(exp_dir: str) -> dict[Path, StageTwoExperimentJsonFormat]:
        paths = ExpLoader.get_paths(exp_dir)
        output = {}
        for path in paths:
            output[Path(path)] = StageTwoExperimentJsonFormat(**json.load(open(path)))
        return output

    @staticmethod
    def stage_one(exp_dir: str) -> dict[Path, ExperimentJsonFormat]:
        paths = ExpLoader.get_paths(exp_dir)
        output = {}
        for path in paths:
            output[Path(path)] = ExperimentJsonFormat(**json.load(open(path)))
        return output


def save_loaded_dict(loaded_dict: LoadedJsonType):
    for file_out, loaded in loaded_dict.items():
        # create the directory if it doesn't exist
        file_out.parent.mkdir(parents=True, exist_ok=True)
        _json = loaded.json(indent=2)
        safe_file_write(str(file_out), _json)
