from typing import Sequence, Union
from cot_transparency.data_models.models import ExperimentJsonFormat, StageTwoExperimentJsonFormat


import json
from glob import glob
from pathlib import Path

from cot_transparency.util import safe_file_write

LoadedJsonType = Union[dict[Path, ExperimentJsonFormat], dict[Path, StageTwoExperimentJsonFormat]]


class ExpLoader:
    @staticmethod
    def get_paths(exp_dir: str, subpath="*/*/*.json") -> list[str]:
        # import ipdb; ipdb.set_trace()
        paths = glob(f"{exp_dir}/{subpath}", recursive=True)
        if paths:
            print(f"Found {len(paths)} json files")
        else:
            raise FileNotFoundError(f"Could not find any json files in {exp_dir}")
        return paths

    @staticmethod
    def get_stage(exp_dir: str, subpath="*/**/*.json") -> int:
        paths = ExpLoader.get_paths(exp_dir, subpath=subpath)
        with open(paths[0]) as f:
            d = json.load(f)
        return int(d["stage"])

    @staticmethod
    def stage_two(
        exp_dir: str,
        final_only: bool = False,
    ) -> dict[Path, StageTwoExperimentJsonFormat]:
        if final_only:
            paths = ExpLoader.get_paths(exp_dir, subpath="*_final/**/*.json")
        else:
            paths = ExpLoader.get_paths(exp_dir, subpath="*/**/*.json")

        # We never want paths inside the stage_one_exp_dir
        paths = [i for i in paths if "stage_one_exp_dir" not in i]

        output = {}
        for path in paths:
            with open(path) as f:
                output[Path(path)] = StageTwoExperimentJsonFormat(**json.load(f))
        return output

    @staticmethod
    def stage_one(exp_dir: str, models: Sequence[str] = []) -> dict[Path, ExperimentJsonFormat]:
        if models:
            paths = []
            for model in models:
                # filter out models that don't match using glob
                subpath = f"*/{model}/*.json"
                paths.extend(ExpLoader.get_paths(exp_dir, subpath=subpath))
        else:
            paths = ExpLoader.get_paths(exp_dir)

        # paths = paths[:4]

        output = {}
        for path in paths:
            with open(path) as f:
                output[Path(path)] = ExperimentJsonFormat(**json.load(f))
        return output


def save_loaded_dict(loaded_dict: LoadedJsonType):
    for file_out, loaded in loaded_dict.items():
        # create the directory if it doesn't exist
        file_out.parent.mkdir(parents=True, exist_ok=True)
        _json = loaded.model_dump_json(indent=2)
        safe_file_write(str(file_out), _json)
