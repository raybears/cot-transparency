from typing import TypedDict
from slist import Slist
from cot_transparency.data_models.models import TaskOutput


stuff: Slist[TaskOutput] = Slist[TaskOutput]()


class _Dict(TypedDict):
    cot_length: int


test_dict: _Dict = {"cot_length": 1}
reveal_type(test_dict)
_dicts = stuff.map(lambda task: {"cot_length": len(task.inference_output.raw_response)})

access = _dicts.map(lambda _dict: _dict["cot_lengt"])
