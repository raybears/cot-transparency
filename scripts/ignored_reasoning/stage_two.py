import random
from pathlib import Path
from typing import List, Literal, Optional, Type

import fire
from git import Sequence
from cot_transparency.data_models.config import CONFIG_MAP
from cot_transparency.data_models.data.task_name_map import task_name_to_data_example

from cot_transparency.data_models.io import ExpLoader
from cot_transparency.data_models.models import (
    ExperimentJsonFormat,
    TraceInfo,
    ModelOutput,
    StageTwoTaskOutput,
    StageTwoTaskSpec,
    TaskOutput,
)
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.transparency.mistakes import (
    CompletePartialCOT,
    FewShotGenerateMistakeFormatter,
)
from cot_transparency.formatters.transparency.s1_baselines import (
    FewShotCOTUnbiasedCompletionNoRoleTameraTFormatter,
)
from cot_transparency.formatters.transparency.util import StageTwoFormatter
from cot_transparency.formatters.transparency.trace_manipulation import get_cot_steps
from cot_transparency.formatters.transparency.util import FullCOTCompletionFormatter, FullCOTFormatter
from cot_transparency.apis.openai.set_key import set_keys_from_env
from cot_transparency.tasks import run_with_caching_stage_two
from cot_transparency.util import get_exp_dir_name
from stage_one import get_valid_stage1_formatters

"""
We take traces generated from stage_one.py and run analysis on them
"""


def filter_cot_by_possible_ends(cot_steps: list[str]) -> list[str]:
    """
    to guard against models that give us way to long COTS becuase it is a completion model reutrn questions again filter
    truncate COTs to the first message that has Human: or Question: or \n\n at the start of the next message
    """

    filtered_steps: list[str] = []
    for i, step in enumerate(cot_steps):
        if i > 2 and step.startswith("\n\nHuman:") or step.startswith("\n\nQuestion:"):
            break
        else:
            filtered_steps.append(step)
            if "answer is" in step:
                break
    return filtered_steps


def get_early_answering_tasks(
    stage_one_output: TaskOutput,
    exp_dir: str,
    temperature: Optional[float] = None,
    n_samples_per_cot: int = 8,
    full_answers_only: bool = False,
) -> List[StageTwoTaskSpec]:
    outputs = []

    original_cot = get_cot_steps(stage_one_output.first_raw_response)
    original_cot = filter_cot_by_possible_ends(original_cot)
    cot_steps = [""] + original_cot  # add empty string to start of COT as we want to get an answer with no reasoning

    rng = random.Random(stage_one_output.task_spec.uid())
    sample_idxs = [0, len(cot_steps) - 1]  # we always do the first and last one to get good aoc numbers
    if not full_answers_only:
        truncated_idxs = rng.sample(range(1, len(cot_steps) - 1), min(n_samples_per_cot, len(cot_steps) - 2))
        sample_idxs.extend(truncated_idxs)

    partial_cot = ""
    for i, cot_step in enumerate(cot_steps):
        partial_cot += cot_step

        if stage_one_output.task_spec.formatter_name == FewShotCOTUnbiasedCompletionNoRoleTameraTFormatter.name():
            Formatter = FullCOTCompletionFormatter
        else:
            Formatter = FullCOTFormatter

        config = stage_one_output.task_spec.inference_config.copy()
        out_file_path: Path = Path(
            f"{exp_dir}/early_answering_final/s1-{stage_one_output.task_spec.formatter_name}/{stage_one_output.task_spec.task_name}/{config.model}/{Formatter.name()}.json"
        )

        if temperature is not None:
            config.temperature = temperature
        config.max_tokens = 30  # code-davinci-002 doesn't return answer unless we set this to 2

        # messages / prompt for stage_two
        messages = FullCOTFormatter.format_example(stage_one_output.task_spec.messages, partial_cot, config.model)

        task = StageTwoTaskSpec(
            stage_one_output=stage_one_output,
            inference_config=config,
            formatter_name=Formatter.name(),
            messages=messages,
            out_file_path=out_file_path,
            n_steps_in_cot_trace=i,
            trace_info=TraceInfo(original_cot=original_cot, complete_modified_cot=partial_cot),
        )

        if i in sample_idxs:
            outputs.append(task)
    return outputs


def create_mistake_task_spec_for_stage_one(
    stage_one_output: TaskOutput,
    exp_dir: str,
    mistake_adding_model: str,
    mistake_adding_temperature: float,
    n_mistake_insertion_points: int,
) -> list[StageTwoTaskSpec]:
    DataExample = task_name_to_data_example(stage_one_output.task_spec.task_name)
    data_example = stage_one_output.task_spec.read_data_example_or_raise(DataExample)
    original_question: str = data_example.get_parsed_input()

    cot_steps = get_cot_steps(stage_one_output.first_raw_response)
    cot_steps = filter_cot_by_possible_ends(cot_steps)

    rng = random.Random(original_question)
    # we don't want to insert mistakes at the first step, because then there is no sentence to make a mistake in
    # as the first one is blank cot
    if len(cot_steps) == 0:
        print("WARNING - skipping task as len(cot_steps) == 0")
        return []

    sample_idxs = [0, len(cot_steps) - 1]  # we always do the first and last one to get good aoc numbers
    if len(cot_steps) > 2:
        mistake_idxs = rng.sample(range(1, len(cot_steps) - 1), min(n_mistake_insertion_points, len(cot_steps) - 2))
        sample_idxs.extend(mistake_idxs)
    sample_idxs = sorted(list(set(sample_idxs)))

    config = CONFIG_MAP[mistake_adding_model].copy()
    config.max_tokens = 100
    config.temperature = mistake_adding_temperature
    config.stop = ["\n", "```"]

    original_model_that_generated_cot = stage_one_output.task_spec.inference_config.model
    path = Path(
        f"{exp_dir}/mistakes_stage1/s1-{stage_one_output.task_spec.formatter_name}/cot-{original_model_that_generated_cot}/{stage_one_output.task_spec.task_name}/"
        f"/mistake-{config.model}/{FewShotGenerateMistakeFormatter.name()}.json"
    )
    out: list[StageTwoTaskSpec] = []
    for i in sample_idxs:
        messages = FewShotGenerateMistakeFormatter.format_example(
            original_question=original_question, sentence=cot_steps[i]
        )
        task_spec = StageTwoTaskSpec(
            stage_one_output=stage_one_output,
            inference_config=config,
            formatter_name=FewShotGenerateMistakeFormatter.name(),
            messages=messages,
            out_file_path=path,
            trace_info=TraceInfo(original_cot=cot_steps, mistake_inserted_idx=i),
        )
        out.append(task_spec)
    return out


def get_mistakes(
    stage_one_outputs: list[TaskOutput],
    exp_dir: str,
    batch: int = 10,
    mistake_adding_model: str = "text-davinci-002",
    mistake_adding_temperature: float = 1.0,
    save_mistake_generating_file_every: int = 50,
    n_mistake_insertion_points: int = 3,
) -> list[StageTwoTaskOutput]:
    # mistakes we need to make call to api to generate the mistake, we may use a different model here
    # e.g Tamera et al use a non RLHF model to generate mistakes
    specs: list[StageTwoTaskSpec] = []
    for stage_one_output in stage_one_outputs:
        specs.extend(
            create_mistake_task_spec_for_stage_one(
                stage_one_output=stage_one_output,
                exp_dir=exp_dir,
                mistake_adding_model=mistake_adding_model,
                mistake_adding_temperature=mistake_adding_temperature,
                n_mistake_insertion_points=n_mistake_insertion_points,
            )
        )

    print(f"1. Generating mistakes using {mistake_adding_model}")
    print("n specs", len(specs))
    print("n unique specs", len(set([spec.uid() for spec in specs])))
    # put this into dataframe

    out = []
    for i in specs:
        d = i.dict()
        d["task_spec"] = i.stage_one_output.task_spec.dict()
        d["messaegs"] = [i.dict() for i in i.messages]
        d["hash"] = i.uid()
        out.append(d)
    import pandas as pd

    df = pd.DataFrame(out)
    df.to_csv("mistake_specs.csv")

    outputs = run_with_caching_stage_two(save_mistake_generating_file_every, batch, specs)

    return filter_mistakes_output(outputs)


def filter_mistakes_output(outputs: Sequence[StageTwoTaskOutput]) -> list[StageTwoTaskOutput]:
    # Discard outputs where there was no output
    filtered_outputs = [output for output in outputs if output.first_parsed_response is not None]
    n_discarded = len(outputs) - len(filtered_outputs)
    print(f"Discarding {n_discarded} outputs where the mistake model output anything")
    outputs = filtered_outputs

    # Discard ouputs where the mistake model didn't deem there to be a reasoning step
    filtered_outputs = [i for i in outputs if "NO_REASONING" not in i.first_parsed_response]  # type: ignore
    n_discarded = len(outputs) - len(filtered_outputs)
    print(f"Discarding {n_discarded} outputs where the mistake model didn't deem there to be a reasoning step")
    return filtered_outputs


def recomplete_cot_with_inserted_mistake(
    generated_mistakes: list[StageTwoTaskOutput],
    exp_dir: str,
    save_completing_with_mistakes_every: int = 50,
    batch: int = 10,
) -> List[StageTwoTaskOutput]:
    mistakes_inserted_at_last_position: list[StageTwoTaskOutput] = []
    specs: list[StageTwoTaskSpec] = []

    for generated_mistake in generated_mistakes:
        if generated_mistake.first_parsed_response is None or "NO_REASONING" in generated_mistake.first_parsed_response:
            print("WARNING - skipping task as NO_REASONING found in the trace passed to the mistake generator")
            continue

        stage_one_output = generated_mistake.task_spec.stage_one_output
        config = stage_one_output.task_spec.inference_config.copy()

        path = Path(
            f"{exp_dir}/mistakes_stage2/s1-{stage_one_output.task_spec.formatter_name}/{stage_one_output.task_spec.task_name}/{config.model}/{CompletePartialCOT.name()}.json"
        )

        trace_info = generated_mistake.task_spec.trace_info
        assert trace_info is not None
        trace_info.sentence_with_mistake = generated_mistake.first_parsed_response

        partial_cot_trace = trace_info.get_trace_upto_mistake()

        messages = CompletePartialCOT.format_example(
            stage_one_output.task_spec.messages, partial_cot_trace, config.model
        )

        task_spec = StageTwoTaskSpec(
            stage_one_output=stage_one_output,
            inference_config=config,
            formatter_name=CompletePartialCOT.name(),
            messages=messages,
            out_file_path=path,
            trace_info=trace_info,
        )

        # if the mistake was the last step in the reasoning trace, then we don't need to complete the COT
        # so just make a task output with no response
        if trace_info.mistake_inserted_idx == len(trace_info.original_cot) - 1:
            output = StageTwoTaskOutput(
                task_spec=task_spec, inference_output=ModelOutput(raw_response="", parsed_response="")
            )
            mistakes_inserted_at_last_position.append(output)
        else:
            specs.append(task_spec)

    print("2. Regenerating COTs with mistakes, note skipping tasks where mistake was last step in COT")
    outputs = run_with_caching_stage_two(save_completing_with_mistakes_every, batch, specs)

    return outputs + mistakes_inserted_at_last_position


def get_best_single_answer_tasks_given_mistakes(
    cots_with_mistakes_outputs: list[StageTwoTaskOutput],
    exp_dir: str,
    temperature: Optional[float] = None,
) -> list[StageTwoTaskSpec]:
    specs: List[StageTwoTaskSpec] = []
    for output in cots_with_mistakes_outputs:
        stage_one_output = output.task_spec.stage_one_output
        config = stage_one_output.task_spec.inference_config.copy()
        config.max_tokens = 30  # code-davinci-002 doesn't return answer unless we set this to greater than 1
        if temperature is not None:
            config.temperature = temperature

        parsed_response: str | None = output.first_parsed_response
        if parsed_response is None:
            print("WARNING - skipping task as parsed_response is None")
            continue

        if stage_one_output.task_spec.formatter_name == FewShotCOTUnbiasedCompletionNoRoleTameraTFormatter.name():
            Formatter = FullCOTCompletionFormatter
        else:
            Formatter = FullCOTFormatter

        path = Path(
            f"{exp_dir}/mistakes_final/s1-{stage_one_output.task_spec.formatter_name}/{stage_one_output.task_spec.task_name}/{config.model}/{Formatter.name()}.json"
        )
        trace_info = output.task_spec.trace_info
        assert trace_info is not None
        trace_info.regenerated_cot_post_mistake = parsed_response
        cot_trace_with_mistake = trace_info.get_complete_modified_cot()

        final_task = StageTwoTaskSpec(
            stage_one_output=output.task_spec.stage_one_output,
            inference_config=config,
            formatter_name=Formatter.name(),
            messages=Formatter.format_example(
                stage_one_output.task_spec.messages, cot_trace_with_mistake, config.model
            ),
            out_file_path=path,
            n_steps_in_cot_trace=len(get_cot_steps(cot_trace_with_mistake)),
            trace_info=trace_info,
        )
        specs.append(final_task)
    return specs


def create_stage_2_tasks(
    stage_1_task_outputs: List[TaskOutput],
    exp_dir: str,
    mistake_model: str,
    temperature: Optional[float] = None,
    tasks: list[Literal["early_answering", "mistakes"]] = ["early_answering", "mistakes"],
    batch: int = 30,
) -> List[StageTwoTaskSpec]:
    tasks_to_run: List[StageTwoTaskSpec] = []
    task_output: TaskOutput
    for task_output in stage_1_task_outputs:
        if "early_answering" in tasks:
            early_answering_tasks = get_early_answering_tasks(task_output, exp_dir, temperature=temperature)
            tasks_to_run.extend(early_answering_tasks)

    if "mistakes" in tasks:
        if "early_answering" not in tasks:
            for task_output in stage_1_task_outputs:
                # we need this as baseline answers that have no mistakes in. We don't need to
                # do this if we are already running early_answering tasks as we will get the
                # baseline answers from those
                early_answering_tasks = get_early_answering_tasks(
                    task_output, exp_dir, temperature=temperature, full_answers_only=True
                )
                tasks_to_run.extend(early_answering_tasks)

        cots_with_mistakes = get_mistakes(
            stage_1_task_outputs, exp_dir, batch=batch, mistake_adding_model=mistake_model
        )
        cots_with_mistakes_outputs = recomplete_cot_with_inserted_mistake(cots_with_mistakes, exp_dir, batch=batch)
        final_tasks = get_best_single_answer_tasks_given_mistakes(
            cots_with_mistakes_outputs, exp_dir, temperature=temperature
        )
        tasks_to_run.extend(final_tasks)

    return tasks_to_run


def get_valid_stage2_formatters(formatters: list[str]):
    VALID_FORMATTERS = StageTwoFormatter.all_formatters()

    for formatter in formatters:
        if formatter not in VALID_FORMATTERS:
            raise ValueError(
                f"stage_two_formatter {formatter} is not valid. Valid formatters are {list(VALID_FORMATTERS.keys())}"
            )

    validated_formatters: list[Type[StageTwoFormatter]] = [VALID_FORMATTERS[formatter] for formatter in formatters]
    return validated_formatters


def filter_stage1_outputs(
    stage1_outputs: dict[Path, ExperimentJsonFormat],
    stage_one_formatters: list[Type[StageOneFormatter]],
    models: list[str],
) -> dict[Path, ExperimentJsonFormat]:
    outputs: dict[Path, ExperimentJsonFormat] = {}
    for k, exp_json in stage1_outputs.items():
        # only need to check the first example as all examples in the same experiment have the same formatter and model
        if len(exp_json.outputs) == 0:
            continue
        task_spec = exp_json.outputs[0].task_spec
        if task_spec.formatter_name in [i.name() for i in stage_one_formatters]:
            if task_spec.inference_config.model in models:
                outputs[k] = exp_json
    return outputs


def main(
    input_exp_dir: str,
    models: Optional[list[str]] = None,
    stage_one_formatters: Optional[list[str]] = None,
    exp_dir: Optional[str] = None,
    experiment_suffix: str = "",
    save_file_every: int = 50,
    batch: int = 30,
    temperature: float = 0.0,
    example_cap: int = 999999999,
    evaluations: list[Literal["early_answering", "mistakes"]] = ["early_answering", "mistakes"],
    mistake_model="text-davinci-002",
    skip_running_traces: bool = False,
):
    if stage_one_formatters is None:
        # don't do any filtering, just use all stage one outputs
        all_formatters = StageOneFormatter.all_formatters().values()
        stage_one_formatters = [i.name() for i in all_formatters if i.is_cot]  # type: ignore
    if models is None:
        # don't do any filtering, just use all models
        models = list(CONFIG_MAP.keys())

    for evaluation in evaluations:
        assert evaluation in ["early_answering", "mistakes"]

    valid_stage_one_formatters = get_valid_stage1_formatters(stage_one_formatters)
    for formatter in valid_stage_one_formatters:
        if not formatter.is_cot:
            raise ValueError("stage two metrics only make sense for COT stage_one_formatters")

    experiment_jsons: dict[Path, ExperimentJsonFormat] = ExpLoader.stage_one(input_exp_dir)
    experiment_jsons = filter_stage1_outputs(experiment_jsons, valid_stage_one_formatters, models)
    if len(experiment_jsons) == 0:
        print("No matching data from stage one found, nothing to run")
        exit(1)
    else:
        print(f"Found {len(experiment_jsons)} matching experiments from stage one")
    exp_dir = get_exp_dir_name(exp_dir, experiment_suffix, sub_dir="stage_two")

    # symlink the stage one experiments (input_exp_dir) into stage_two exp_dir
    # as stage_one_exp_dir
    # so we can easily see what stage one experiments were used to generate stage two
    Path(f"{exp_dir}/stage_one_exp_dir")
    # if not stage_one_exp_dir.exists():
    #     Path(exp_dir).mkdir(parents=True, exist_ok=True)
    #     stage_one_exp_dir.symlink_to(Path(input_exp_dir).absolute())
    # else:
    #     assert stage_one_exp_dir.resolve() == Path(input_exp_dir).absolute()

    # create flat list of task outputs
    stage_one_outputs_to_use: List[TaskOutput] = []
    for experiment_json in experiment_jsons.values():
        stage_one_outputs = experiment_json.outputs
        # sort based on task_hash
        stage_one_outputs = sorted(stage_one_outputs, key=lambda x: (x.task_spec.repeat_idx, x.task_spec.task_hash))
        stage_one_outputs = stage_one_outputs[:example_cap]
        stage_one_outputs_to_use.extend(stage_one_outputs)

    stage_2_tasks = create_stage_2_tasks(
        stage_one_outputs_to_use,
        exp_dir,
        temperature=temperature,
        tasks=evaluations,
        batch=batch,
        mistake_model=mistake_model,
    )

    if not skip_running_traces:
        run_with_caching_stage_two(save_file_every, batch, stage_2_tasks)


if __name__ == "__main__":
    set_keys_from_env()
    fire.Fire(main)
