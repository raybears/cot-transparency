from slist import Slist

from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.interventions.few_shots_loading import get_correct_cots
from cot_transparency.openai_utils.finetune import FinetuneSample, FineTuneParams, run_finetune


def fine_tune_with_gpt4_cots():
    cots: Slist[TaskOutput] = get_correct_cots()
    messages = [FinetuneSample.from_task_output(task) for task in cots]
    params = FineTuneParams(model="gpt-3.5-turbo")
    _id = run_finetune(params=params, samples=messages)
