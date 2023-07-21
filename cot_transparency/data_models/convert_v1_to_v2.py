from glob import glob
import json
from pathlib import Path
from cot_transparency.data_models.io import save_loaded_dict
from cot_transparency.data_models.models_v1 import ExperimentJsonFormat as ExperimentJsonFormatV1
from cot_transparency.data_models.models_v1 import TaskOutput as TaskOutputV1

from cot_transparency.data_models.models_v2 import (
    ExperimentJsonFormat,
    TaskOutput,
    TaskSpec,
    StageTwoTaskOutput,
    StageTwoTaskSpec,
)
import fire


def load_jsons(exp_dir: str) -> dict[Path, ExperimentJsonFormatV1]:
    loaded_dict: dict[Path, ExperimentJsonFormatV1] = {}

    paths = glob(f"{exp_dir}/*/*/*.json", recursive=True)
    print(f"Found {len(paths)} json files")
    for path in paths:
        _dict = json.load(open(path))
        loaded_dict[Path(path)] = ExperimentJsonFormatV1(**_dict)
    return loaded_dict


def stage_1_output_convert(output: TaskOutputV1, new_path: Path) -> TaskOutput:
    task_spec = TaskSpec(
        task_hash=output.task_hash,
        messages=output.prompt,
        model_config=output.config,
        formatter_name=output.formatter_name,
        task_name=output.task_name,
        out_file_path=new_path,
        ground_truth=output.ground_truth,
        biased_ans=output.biased_ans,
    )
    assert len(output.model_output) == 1, "We only support one model output per task"
    new_output = TaskOutput(
        task_spec=task_spec,
        model_output=output.model_output[0],
    )
    return new_output


def stage_2_output_convert(output: TaskOutputV1, stage_one_output: TaskOutputV1, new_path: Path) -> StageTwoTaskOutput:
    new_stage_one_output: TaskOutput = stage_1_output_convert(stage_one_output, new_path)

    task_spec = StageTwoTaskSpec(
        messages=output.prompt,
        model_config=output.config,
        formatter_name=output.formatter_name,
        out_file_path=new_path,
        step_in_cot_trace=output.step_in_cot_trace,
        stage_one_output=new_stage_one_output,
    )
    assert len(output.model_output) == 1, "We only support one model output per task"
    new_output = StageTwoTaskOutput(
        task_spec=task_spec,
        model_output=output.model_output[0],
    )
    return new_output


def convert_v1_to_v2(exp_dir: str, new_exp_dir: str):
    loaded = load_jsons(exp_dir)

    new_items: dict[Path, ExperimentJsonFormat] = {}
    for path, exp in loaded.items():
        new_path = Path(str(path).replace(exp_dir, new_exp_dir))
        new_outputs = []
        output: TaskOutputV1
        for output in exp.outputs:
            new_output = stage_1_output_convert(output, new_path)
            new_outputs.append(new_output)

        new_items[new_path] = ExperimentJsonFormat(outputs=new_outputs)

    save_loaded_dict(new_items)


if __name__ == "__main__":
    fire.Fire(convert_v1_to_v2)
