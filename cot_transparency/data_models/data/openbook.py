from cot_transparency.data_models.data.arc import ArcExample, load_arc

# same format as arc


def test() -> list[ArcExample]:
    dev_path = "./data/openbook_qa/test.jsonl"
    return load_arc(dev_path)
