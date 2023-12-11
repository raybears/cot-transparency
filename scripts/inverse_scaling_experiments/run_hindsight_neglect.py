from typing import Mapping

from slist import Slist

from cot_transparency.apis.base import ModelCaller
from cot_transparency.data_models.data import InverseScalingTask
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.inverse_scaling.no_few_shot import ClearFewShotsCOT, ClearFewShotsThinkStepByStepCOT
from cot_transparency.streaming.stage_one_stream import stage_one_stream


async def run_hindsight_neglect_for_models(
    caller: ModelCaller, models: list[str], example_cap: int = 600
) -> Mapping[str, float]:
    """Returns 1-accuracy for each model"""
    stage_one_obs = stage_one_stream(
        formatters=[ClearFewShotsCOT().name(), ClearFewShotsThinkStepByStepCOT().name()],
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
    results_filtered = results.filter(lambda x: x.first_parsed_response is not None)
    # group by model
    one_minus_accuracy = results_filtered.group_by(lambda x: x.task_spec.inference_config.model).map(
        lambda group: group.map_values(lambda v: 1 - v.map(lambda task: task.is_correct).average_or_raise())
    )
    return one_minus_accuracy.to_dict()
