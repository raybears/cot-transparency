from pathlib import Path
from typing import Sequence

from slist import Slist

from cot_transparency.data_models.models import TaskOutput, ExperimentJsonFormat
from cot_transparency.tasks import read_done_experiment


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
