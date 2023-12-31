from typing import Optional, Sequence, Type

from slist import Slist

from cot_transparency.data_models.data.task_name_map import task_name_to_data_example
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.interventions.intervention import Intervention
from scripts.multi_accuracy import AccuracyInput, PlotInfo, accuracy_outputs_from_inputs
from scripts.simple_formatter_names import INTERVENTION_TO_SIMPLE_NAME


def pick_random_ans(
    task: TaskOutput,
) -> AccuracyInput:
    task_type = task.task_spec.task_name
    task_class = task_name_to_data_example(task_type)
    read: DataExampleBase = task.task_spec.read_data_example_or_raise(task_class)
    read._get_options()
    lettered = read.get_lettered_options()
    letter = Slist(lettered).map(lambda x: x.indicator).sample(1, seed=task.task_spec.task_hash).first_or_raise()
    biased_ans = task.task_spec.biased_ans
    assert biased_ans is not None

    return AccuracyInput(ground_truth=biased_ans, predicted=letter)


def random_chance_matching_answer_plot_dots(
    all_tasks: Sequence[TaskOutput],
    model: str,
    formatter: Type[StageOneFormatter] = ZeroShotCOTUnbiasedFormatter,
    name_override: Optional[str] = None,
    for_task: Sequence[str] = [],
) -> PlotInfo:
    intervention_name = None
    filtered: Slist[TaskOutput] = (
        Slist(all_tasks)
        .filter(lambda task: intervention_name == task.task_spec.intervention_name)
        .filter(lambda task: task.task_spec.formatter_name == formatter.name())
        .filter(lambda task: task.task_spec.inference_config.model == model)
        .filter(lambda task: task.task_spec.task_name in for_task if for_task else True)
    )
    assert filtered, f"Intervention None has no tasks in {formatter.name()}"
    transformed = Slist(filtered).map(pick_random_ans)
    outputs = accuracy_outputs_from_inputs(transformed)
    return PlotInfo(acc=outputs, name=name_override or "Random chance")


def matching_user_answer_plot_info(
    all_tasks: Sequence[TaskOutput],
    intervention: Optional[Type[Intervention]] = None,
    for_formatters: Sequence[Type[StageOneFormatter]] = [],
    model: Optional[str] = None,
    for_task: Sequence[str] = [],
    name_override: Optional[str] = None,
    distinct_qns: bool = True,
) -> PlotInfo:
    intervention_name: str | None = intervention.name() if intervention else None
    formatters_names: set[str] = {f.name() for f in for_formatters}
    filtered: Slist[TaskOutput] = (
        Slist(all_tasks)
        .filter(lambda task: intervention_name == task.task_spec.intervention_name if intervention_name else True)
        .filter(lambda task: task.task_spec.formatter_name in formatters_names if formatters_names else True)
        .filter(lambda task: task.task_spec.inference_config.model == model if model else True)
        .filter(lambda task: task.task_spec.task_name in for_task if for_task else True)
    )
    if distinct_qns:
        filtered = filtered.distinct_by(lambda task: task.task_spec.task_hash)
    assert filtered, f"Intervention {intervention_name} has no tasks in {for_formatters} for {model}"
    transformed = (
        Slist(filtered)
        .map(
            # We want to measure how often the model's answer matches the user's answer
            lambda x: AccuracyInput(ground_truth=x.task_spec.biased_ans, predicted=x.first_parsed_response)
            if x.first_parsed_response is not None and x.task_spec.biased_ans is not None
            else None
        )
        .flatten_option()
    )
    outputs = accuracy_outputs_from_inputs(transformed)
    retrieved_simple_name: str | None = INTERVENTION_TO_SIMPLE_NAME.get(intervention, None)
    name: str = name_override or retrieved_simple_name or intervention_name or "No intervention, biased context"
    return PlotInfo(acc=outputs, name=name)
