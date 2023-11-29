from collections import Counter
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel

from slist import Slist

from cot_transparency.data_models.example_base import DataExampleBase, MultipleChoiceAnswer


class RefusalExample(DataExampleBase):
    question: str
    category: str

    def _get_question(self) -> str:
        return self.question.strip()

    def _get_options(self) -> list[str]:
        return [""]

    @property
    def _ground_truth(self) -> MultipleChoiceAnswer:
        return "A"  # dummy answer to make the code work


def load_data(questions_per_category: int = 20) -> Slist[RefusalExample]:
    path = "./data/refusal/filtered_questions.jsonl"
    data = read_jsonl_file_into_basemodel(path, RefusalExample)

    outputs = Slist()
    questions_per_category_counter = Counter()
    for i, example in enumerate(data):
        if questions_per_category_counter[example.category] <= questions_per_category:
            outputs.append(example)

    return outputs


if __name__ == "__main__":
    outputs = load_data(5)
