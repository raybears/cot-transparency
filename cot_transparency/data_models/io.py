from typing import Union
from cot_transparency.data_models.models_v2 import ExperimentJsonFormat, StageTwoExperimentJsonFormat


import json
from glob import glob
from pathlib import Path

from cot_transparency.util import safe_file_write

LoadedJsonType = Union[dict[Path, ExperimentJsonFormat], dict[Path, StageTwoExperimentJsonFormat]]


def load_jsons(exp_dir: str) -> tuple[LoadedJsonType, bool]:
    paths = glob(f"{exp_dir}/*/*/*.json", recursive=True)
    if paths:
        print(f"Found {len(paths)} json files")
        is_stage_two = json.load(open(paths[0]))["stage"] == 2
        if is_stage_two:
            return _load_stage2_jsons(paths), True
        else:
            return _load_stage1_jsons(paths), False
    else:
        raise ValueError(f"No json files found in {exp_dir}")


def _load_stage1_jsons(paths: list[str]) -> dict[Path, ExperimentJsonFormat]:
    output = {}
    for path in paths:
        output[Path(path)] = ExperimentJsonFormat(**json.load(open(path)))
    return output


def _load_stage2_jsons(paths: list[str]) -> dict[Path, StageTwoExperimentJsonFormat]:
    output = {}
    for path in paths:
        output[Path(path)] = StageTwoExperimentJsonFormat(**json.load(open(path)))
    return output


def save_loaded_dict(loaded_dict: LoadedJsonType):
    for file_out, loaded in loaded_dict.items():
        # create the directory if it doesn't exist
        file_out.parent.mkdir(parents=True, exist_ok=True)
        _json = loaded.json(indent=2)
        safe_file_write(str(file_out), _json)
