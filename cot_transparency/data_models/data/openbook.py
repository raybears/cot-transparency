from pathlib import Path

from cot_transparency.data_models.data.arc import ArcExample
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel


def test() -> list[ArcExample]:
    dev_path = Path("./data/openbook_qa/test.jsonl")
    return read_jsonl_file_into_basemodel(dev_path, ArcExample)


def openbook_train() -> list[ArcExample]:
    path = Path("./data/openbook_qa/train.jsonl")
    return read_jsonl_file_into_basemodel(path, ArcExample)
