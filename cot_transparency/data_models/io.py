import itertools
import json
from glob import glob
from pathlib import Path
from typing import Optional, Sequence, Union

from slist import Slist

from cot_transparency.data_models.models import (
    ExperimentJsonFormat,
    StageTwoExperimentJsonFormat,
    StageTwoTaskOutput,
    TaskOutput,
)
from cot_transparency.json_utils.read_write import safe_file_write

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
    def stage_one(
        exp_dir: str,
        models: Optional[Sequence[str]] = None,
        task_names: Optional[Sequence[str]] = None,
    ) -> dict[Path, ExperimentJsonFormat]:
        if models is None:
            models = ["*"]
        if task_names is None:
            task_names = ["*"]

        # get the cross product of models and task_names
        combos = list(itertools.product(models, task_names))
        paths = []
        for model, task_name in combos:
            # filter out models that don't match using glob
            subpath = f"{task_name}/{model}/*.json"
            paths.extend(ExpLoader.get_paths(exp_dir, subpath=subpath))

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


def read_done_experiment(out_file_path: Path) -> ExperimentJsonFormat:
    # read in the json file
    if out_file_path.exists():
        with open(out_file_path) as f:
            _dict = json.load(f)
            if _dict["stage"] == 2:
                raise ValueError(
                    "This looks like a stage two experiment but you are trying to"
                    "read it as stage one, maybe use read_whole_exp_dir_s2 instead"
                )
            return ExperimentJsonFormat(**_dict)
    else:
        return ExperimentJsonFormat(outputs=[])


def read_whole_exp_dir(exp_dir: str) -> Slist[TaskOutput]:
    """
    find formatter names from the exp_dir
    exp_dir/task_name/model/formatter_name.json
    """
    json_files = glob(f"{exp_dir}/*/*/*.json")
    read: Slist[TaskOutput] = (
        Slist(json_files).map(Path).map(read_done_experiment).map(lambda exp: exp.outputs).flatten_list()
    )
    print(f"Read {len(read)} tasks from {exp_dir}")
    return read


def read_whole_exp_dir_s2(exp_dir: str) -> Slist[StageTwoTaskOutput]:
    loaded_dict = ExpLoader.stage_two(exp_dir=exp_dir)
    outputs: Slist[StageTwoTaskOutput] = Slist()
    for task_output in loaded_dict.values():
        outputs.extend(task_output.outputs)
    print(f"Read {len(outputs)} tasks from {exp_dir}")
    return outputs


def get_loaded_dict_stage2(
    paths: set[Path],
) -> dict[Path, StageTwoExperimentJsonFormat]:
    # work out which tasks we have already done
    loaded_dict: dict[Path, StageTwoExperimentJsonFormat] = {}
    for path in paths:
        if path.exists():
            with open(path) as f:
                done_exp = StageTwoExperimentJsonFormat(**json.load(f))
            # Override to ensure bwds compat with some old exps that had the wrong stage
            done_exp.stage = 2
            loaded_dict[path] = done_exp
        else:
            loaded_dict[path] = StageTwoExperimentJsonFormat(outputs=[])
    return loaded_dict


def read_all_for_selections(
    exp_dirs: Sequence[Path],
    formatters: Sequence[str],
    models: Sequence[str],
    tasks: Sequence[str],
    interventions: Sequence[str | None] = [],
) -> Slist[TaskOutput]:
    # More efficient than to load all the experiments in a directory
    task_outputs: Slist[TaskOutput] = Slist()
    # Add None to interventions if empty
    interventions_none = [None] if not interventions else interventions
    for exp_dir in exp_dirs:
        for formatter in formatters:
            for task in tasks:
                for model in models:
                    for intervention in interventions_none:
                        if intervention is None:
                            path = exp_dir / f"{task}/{model}/{formatter}.json"
                        else:
                            path = exp_dir / f"{task}/{model}/{formatter}_and_{intervention}.json"
                        experiment: ExperimentJsonFormat = read_done_experiment(path)
                        task_outputs.extend(experiment.outputs)
    return task_outputs
