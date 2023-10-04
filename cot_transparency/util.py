import datetime
import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, TypeVar, Sequence

from slist import Slist

from cot_transparency.data_models.models import TaskOutput, ExperimentJsonFormat
from cot_transparency.tasks import read_done_experiment

A = TypeVar("A")


def get_exp_dir_name(
    exp_dir: Optional[str] = None,
    experiment_suffix: Optional[str] = None,
    sub_dir: Optional[str] = None,
    root: str = "./experiments",
):
    if exp_dir is None:
        root = "./experiments"
        if sub_dir:
            root = os.path.join(root, sub_dir)
        now = datetime.datetime.now().strftime("%Y%m%d")
        exp_dir = os.path.join(root, now)
    if experiment_suffix:
        exp_dir += f"_{experiment_suffix}"
    return exp_dir


def setup_logger(name: str, level: int = logging.WARNING):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def safe_file_write(filename: str, data: str):
    dir_name = os.path.dirname(filename)
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as f:
        f.write(data)
        temp_name = f.name
    try:
        os.rename(temp_name, filename)
    except Exception as e:
        os.remove(temp_name)
        raise e


def deterministic_hash(something: str) -> str:
    return hashlib.sha1(something.encode()).hexdigest()


def assert_not_none(x: A | None) -> A:
    assert x is not None, "Expected not None"
    return x


def read_all_for_selections(
    exp_dirs: Sequence[Path], formatters: Sequence[str], models: Sequence[str], tasks: Sequence[str]
) -> Slist[TaskOutput]:
    # More efficient than to load all the experiments in a directory
    task_outputs: Slist[TaskOutput] = Slist()
    for exp_dir in exp_dirs:
        for formatter in formatters:
            for task in tasks:
                for model in models:
                    path = exp_dir / f"{task}/{model}/{formatter}.json"
                    experiment: ExperimentJsonFormat = read_done_experiment(path)
                    assert experiment.outputs, f"Experiment {path} has no outputs"
                    task_outputs.extend(experiment.outputs)
    return task_outputs
