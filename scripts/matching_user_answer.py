from typing import Optional, Type, Sequence

from slist import Slist

from cot_transparency.data_models.data import task_name_to_data_example
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.interventions.intervention import Intervention
from scripts.multi_accuracy import PlotDots, AccuracyInput, accuracy_outputs_from_inputs
from scripts.simple_formatter_names import INTERVENTION_TO_SIMPLE_NAME


def pick_random_ans(
    task: TaskOutput,
) -> AccuracyInput:
    task_type = task.task_spec.task_name
    task_class = task_name_to_data_example(task_type)
    read: DataExampleBase = task.task_spec.read_data_example_or_raise(task_class)
    options = read._get_options()
    lettered = read._get_lettered_options(options=options)
    letter = Slist(lettered).map(lambda x: x.letter).sample(1, seed=task.task_spec.task_hash).first_or_raise()
    biased_ans = task.task_spec.biased_ans
    assert biased_ans is not None

    return AccuracyInput(ground_truth=biased_ans, predicted=letter)


def baseline_matching_answer_plot_dots(
    all_tasks: Sequence[TaskOutput],
    model: str,
    formatter: Type[StageOneFormatter] = ZeroShotCOTUnbiasedFormatter,
    name_override: Optional[str] = None,
) -> PlotDots:
    intervention_name = None
    filtered: Slist[TaskOutput] = (
        Slist(all_tasks)
        .filter(lambda task: intervention_name == task.task_spec.intervention_name)
        .filter(lambda task: task.task_spec.formatter_name == formatter.name())
        .filter(lambda task: task.task_spec.model_config.model == model)
    )
    assert filtered, f"Intervention None has no tasks in {formatter.name()}"
    transformed = Slist(filtered).map(pick_random_ans)
    outputs = accuracy_outputs_from_inputs(transformed)
    return PlotDots(acc=outputs, name=name_override or "Random chance")


def matching_user_answer_plot_dots(
    intervention: Optional[Type[Intervention]],
    all_tasks: Sequence[TaskOutput],
    for_formatters: Sequence[Type[StageOneFormatter]],
    model: str,
    name_override: Optional[str] = None,
) -> PlotDots:
    intervention_name: str | None = intervention.name() if intervention else None
    formatters_names: set[str] = {f.name() for f in for_formatters}
    filtered: Slist[TaskOutput] = (
        Slist(all_tasks)
        .filter(lambda task: intervention_name == task.task_spec.intervention_name)
        .filter(lambda task: task.task_spec.formatter_name in formatters_names)
        .filter(lambda task: task.task_spec.model_config.model == model)
    )
    assert filtered, f"Intervention {intervention_name} has no tasks in {for_formatters}"
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
    return PlotDots(acc=outputs, name=name)
