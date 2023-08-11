from slist import Slist

from cot_transparency.data_models.io import ExpLoader
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.model_apis import Prompt
from scripts.biased_wrong_ans import (
    BiasedWrongSplit,
    filter_for_correct_cot,
    paired_sample_few_shots_cot,
)

jsons = ExpLoader.stage_one("experiments/bad_cot")


jsons_tasks: Slist[TaskOutput] = Slist(jsons.values()).map(lambda x: x.outputs).flatten_list()  # type: ignore
selected_formatter = "ZeroShotCOTUnbiasedFormatter"
print(f"Number of jsons: {len(jsons_tasks)}")
all_correct_cot: BiasedWrongSplit = filter_for_correct_cot(jsons_tasks, selected_formatter).balance()
few_shots_to_sample = Slist(all_correct_cot.wrong_biased) + Slist(all_correct_cot.correct_biased)

consistency_few_shot: Prompt = paired_sample_few_shots_cot(few_shots_to_sample, seed="1", n=5)
