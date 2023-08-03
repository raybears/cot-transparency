from typing import Union
from cot_transparency.data_models.models import ExperimentJsonFormat, StageTwoExperimentJsonFormat


import json
from glob import glob
from pathlib import Path

from cot_transparency.util import safe_file_write

LoadedJsonType = Union[dict[Path, ExperimentJsonFormat], dict[Path, StageTwoExperimentJsonFormat]]


class ExpLoader:
    @staticmethod
    def get_paths(exp_dir: str, subpath="*/*/*.json") -> list[str]:
        paths = glob(f"{exp_dir}/{subpath}", recursive=True)
        if paths:
            print(f"Found {len(paths)} json files")
        else:
            raise FileNotFoundError(f"Could not find any json files in {exp_dir}")
        return paths

    @staticmethod
    def get_stage(exp_dir: str) -> int:
        paths = ExpLoader.get_paths(exp_dir)
        with open(paths[0]) as f:
            first_exp = json.load(f)
        return int(first_exp["stage"])

    @staticmethod
    def stage_two(exp_dir: str) -> dict[Path, StageTwoExperimentJsonFormat]:
        paths = ExpLoader.get_paths(exp_dir)
        output = {}
        for path in paths:
            with open(path) as f:
                output[Path(path)] = StageTwoExperimentJsonFormat(**json.load(f))
        return output

    @staticmethod
    def stage_one(exp_dir: str) -> dict[Path, ExperimentJsonFormat]:
        paths = ExpLoader.get_paths(exp_dir)
        output = {}
        for path in paths:
            with open(path) as f:
                output[Path(path)] = ExperimentJsonFormat(**json.load(f))
        return output

    @staticmethod
    def stage_two_mistake_generation(exp_dir: str) -> dict[tuple[str, str, str], StageTwoExperimentJsonFormat]:
        # format is exp_dir/mistake_generation/cot-model/task/mistake-model/formatter.json
        paths = ExpLoader.get_paths(exp_dir, subpath="./mistake_generation/*/*/*.json")
        output = {}
        for path in paths:
            with open(path) as f:
                _path = Path(path)
                output[(_path.parent.parent.parent.name, _path.parent.name, _path.name)] = StageTwoExperimentJsonFormat(
                    **json.load(f)
                )
        return output

    @staticmethod
    def stage_two_cots_with_mistakes(exp_dir: str) -> dict[Path, StageTwoExperimentJsonFormat]:
        paths = ExpLoader.get_paths(exp_dir, subpath="*/*/cots_with_mistakes/*.json")
        output = {}
        for path in paths:
            with open(path) as f:
                output[Path(path)] = StageTwoExperimentJsonFormat(**json.load(f))
        return output


def save_loaded_dict(loaded_dict: LoadedJsonType):
    for file_out, loaded in loaded_dict.items():
        # create the directory if it doesn't exist
        file_out.parent.mkdir(parents=True, exist_ok=True)
        _json = loaded.json(indent=2)
        safe_file_write(str(file_out), _json)
