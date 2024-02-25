from slist import Slist

from cot_transparency.apis.base import CachedPerModelCaller, ModelCaller
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.data import InverseScalingTask
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.data_models.pd_utils import DataRow
from cot_transparency.formatters.inverse_scaling.no_few_shot import (
    ClearFewShotsCOT,
    ClearFewShotsCOTVariant,
    ClearFewShotsThinkStepByStepCOT,
    ClearFewShotsThinkStepByStepCOTVariant,
    FewShotsSpecified,
    MoreNonSpuriousFewShots,
    OnlyNonSpuriousFewShots,
    RemoveInverseScalingFewShotsCOT,
    HindsightNeglectPointOutCOT,
)
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.automated_answer_parsing.answer_parsing_example import answer_finding_step
from scripts.evil_grid_exp.message_to_csv_display import messages_to_str


# def hindsight_neglect_to_data_row(
#     x: TaskOutput) -> DataRow:
#     return DataRow(
#         model=x.task_spec.inference_config.model,
#         is_cot=True,
#         matches_bias=1 - x.is_correct,
#         is_correct=x.is_correct,
#         task="6) Spurious Few Shot: Hindsight",
#         bias_name="6) Spurious Few Shot: Hindsight",
#     )


def hindsight_to_data_row(task: TaskOutput) -> DataRow:
    x = task
    # formatter = x.get_task_spec().formatter_name
    # biased_ans = x.get_task_spec().biased_ans
    model = x.get_task_spec().inference_config.model
    return DataRow(
        model=model,
        model_type=None,
        bias_name="6) Spurious Few Shot: Hindsight",
        task="6) Spurious Few Shot: Hindsight",
        unbiased_question=x.get_task_spec().get_data_example_obj().get_parsed_input(),
        biased_question=messages_to_str(x.get_task_spec().messages),
        question_id=x.get_task_spec().get_data_example_obj().hash(),
        ground_truth=x.get_task_spec().ground_truth,
        biased_ans="TODO",
        raw_response=x.inference_output.raw_response or "FAILED_SAMPLE",
        parsed_response=x.inference_output.parsed_response or "FAILED_SAMPLE",
        parsed_ans_matches_bias=not x.is_correct,
        is_cot=True,
        is_correct=x.is_correct,
    )


async def run_hindsight_neglect_for_models(
    answer_parsing_config: OpenaiInferenceConfig,
    answer_parsing_caller: CachedPerModelCaller,
    caller: ModelCaller,
    models: list[str],
    example_cap: int = 600,
) -> Slist[TaskOutput]:
    """Returns 1-accuracy for each model"""
    stage_one_obs = stage_one_stream(
        formatters=[
            ClearFewShotsCOT().name(),
            ClearFewShotsCOTVariant().name(),
            ClearFewShotsThinkStepByStepCOT().name(),
            ClearFewShotsThinkStepByStepCOTVariant().name(),
        ],
        tasks=[InverseScalingTask.hindsight_neglect],
        example_cap=example_cap,
        num_tries=1,
        raise_after_retries=False,
        temperature=0.0,
        caller=caller,
        batch=40,
        models=models,
    ).map_blocking_par(
        # annoyingly need to parse out because the model returns non-standard responses
        lambda task: answer_finding_step(task, answer_parsing_caller, config=answer_parsing_config)
    )

    results: Slist[TaskOutput] = await stage_one_obs.to_slist()

    write_jsonl_file_from_basemodel("hindsight_neglect.jsonl", results)
    # sort by task hash so that sampling is deterministic
    results_filtered = results.filter(lambda x: x.first_parsed_response is not None).sort_by(
        lambda x: x.task_spec.task_hash
    )
    # group by model

    # return results_filtered

    # out = results_filtered.map(hindsight_to_data_row)

    return results_filtered


async def run_hindsight_neglect_few_shots_specified(
    answer_parsing_config: OpenaiInferenceConfig,
    answer_parsing_caller: CachedPerModelCaller,
    caller: ModelCaller,
    models: list[str],
    example_cap: int = 600,
) -> Slist[TaskOutput]:
    """Returns 1-accuracy for each model"""
    stage_one_obs = stage_one_stream(
        formatters=[
            FewShotsSpecified().name(),
        ],
        tasks=[InverseScalingTask.hindsight_neglect],
        example_cap=example_cap,
        num_tries=1,
        raise_after_retries=False,
        temperature=0.0,
        caller=caller,
        batch=40,
        models=models,
    ).map_blocking_par(
        # annoyingly need to parse out because the model returns non-standard responses
        lambda task: answer_finding_step(task, answer_parsing_caller, config=answer_parsing_config)
    )

    results: Slist[TaskOutput] = await stage_one_obs.to_slist()

    write_jsonl_file_from_basemodel("hindsight_neglect_specified.jsonl", results)
    # sort by task hash so that sampling is deterministic
    results_filtered = results.filter(lambda x: x.first_parsed_response is not None).sort_by(
        lambda x: x.task_spec.task_hash
    )
    # group by model

    # return results_filtered

    # out = results_filtered.map(hindsight_to_data_row)

    return results_filtered


async def run_hindsight_neglect_better_few_shots(
    caller: ModelCaller, models: list[str], example_cap: int = 600
) -> Slist[TaskOutput]:
    """Returns 1-accuracy for each model"""
    stage_one_obs = stage_one_stream(
        formatters=[MoreNonSpuriousFewShots().name()],
        tasks=[InverseScalingTask.hindsight_neglect],
        example_cap=example_cap,
        num_tries=1,
        raise_after_retries=False,
        temperature=0.0,
        caller=caller,
        batch=40,
        models=models,
    )

    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    write_jsonl_file_from_basemodel("hindsight_neglect_more_non_spurious.jsonl", results)
    results_filtered = (
        results.filter(lambda x: x.first_parsed_response is not None).sort_by(lambda x: x.task_spec.task_hash)
        # .map(hindsight_to_data_row)
    )
    # group by model
    return results_filtered


async def run_hindsight_neglect_only_non_spurious(
    caller: ModelCaller, models: list[str], example_cap: int = 600
) -> Slist[TaskOutput]:
    """Returns 1-accuracy for each model"""
    stage_one_obs = stage_one_stream(
        formatters=[OnlyNonSpuriousFewShots().name()],
        tasks=[InverseScalingTask.hindsight_neglect],
        example_cap=example_cap,
        num_tries=1,
        raise_after_retries=False,
        temperature=0.0,
        caller=caller,
        batch=40,
        models=models,
    )

    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    write_jsonl_file_from_basemodel("hindsight_neglect_only_non_spurious.jsonl", results)
    results_filtered = (
        results.filter(lambda x: x.first_parsed_response is not None).sort_by(lambda x: x.task_spec.task_hash)
        # .map(hindsight_to_data_row)
    )
    # group by model
    return results_filtered


async def run_hindsight_neglect_zero_shot(
    caller: ModelCaller, models: list[str], example_cap: int = 600
) -> Slist[DataRow]:
    """Returns 1-accuracy for each model"""
    stage_one_obs = stage_one_stream(
        formatters=[RemoveInverseScalingFewShotsCOT().name()],
        tasks=[InverseScalingTask.hindsight_neglect],
        example_cap=example_cap,
        num_tries=1,
        raise_after_retries=False,
        temperature=0.0,
        caller=caller,
        batch=40,
        models=models,
    )

    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    write_jsonl_file_from_basemodel("hindsight_neglect_zeroshot.jsonl", results)
    results_filtered = (
        results.filter(lambda x: x.first_parsed_response is not None)
        .sort_by(lambda x: x.task_spec.task_hash)
        .map(hindsight_to_data_row)
    )
    # group by model
    return results_filtered


async def run_hindsight_neglect_zeroshot_better_instruction(
    caller: ModelCaller, models: list[str], example_cap: int = 600
) -> Slist[DataRow]:
    """Returns 1-accuracy for each model"""
    stage_one_obs = stage_one_stream(
        formatters=[HindsightNeglectPointOutCOT().name()],
        tasks=[InverseScalingTask.hindsight_neglect],
        example_cap=example_cap,
        num_tries=1,
        raise_after_retries=False,
        temperature=0.0,
        caller=caller,
        batch=40,
        models=models,
    )

    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    results_filtered = (
        results.filter(lambda x: x.first_parsed_response is not None)
        .sort_by(lambda x: x.task_spec.task_hash)
        .map(hindsight_to_data_row)
    )
    # write_jsonl_file_from_basemodel("hindsight_neglect_base.jsonl", results_filtered)
    # group by model
    return results_filtered


# async def hindsight_neglect_non_spurious_baseline(
#     caller: ModelCaller, models: list[str], example_cap: int = 600
# ) -> Slist[DataRow]:
#     """Returns 1-accuracy for each model"""
#     stage_one_obs = stage_one_stream(
#         formatters=[NonSpuriousFewShotHindsightCOT().name()],
#         tasks=[InverseScalingTask.hindsight_neglect],
#         example_cap=example_cap,
#         num_tries=1,
#         raise_after_retries=False,
#         temperature=0.0,
#         caller=caller,
#         batch=40,
#         models=models,
#     )

#     results: Slist[TaskOutput] = await stage_one_obs.to_slist()
#     results_filtered = results.filter(lambda x: x.first_parsed_response is not None)
#     write_jsonl_file_from_basemodel("hindsight_neglect_base_non_spuriou.jsonl", results_filtered)
#     # group by model

#     out = results_filtered.map(
#         lambda x: DataRow(
#             model=x.task_spec.inference_config.model,
#             is_cot=True,
#             matches_bias=1 - x.is_correct,
#             is_correct=x.is_correct,
#             task="zzzz Hindsight Neglect Non Spurious Fewshot",
#             bias_name="zzzz Hindsight Neglect Non Spurious Fewshot",
#         )
#     )

#     return out
