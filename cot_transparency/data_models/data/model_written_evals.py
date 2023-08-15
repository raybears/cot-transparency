import re
from pathlib import Path

from slist import Slist

from cot_transparency.data_models.data.strip_brackets import (
    extract_answer_from_brackets,
    raise_if_not_multiple_choice_answer,
)
from cot_transparency.data_models.example_base import DataExampleBase, MultipleChoiceAnswer
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel


class ModelWrittenEvalsExample(DataExampleBase):
    # https://huggingface.co/datasets/Anthropic/model-written-evals/viewer/Anthropic--model-written-evals
    # Note that this dataset is already biased, there isn't an unbiased version of it
    # We could strip out the bias to make it unbiased, but we don't need to do that for now
    question: str
    answer_matching_behavior: str

    @property
    def biased_ans(self) -> MultipleChoiceAnswer:
        ans = extract_answer_from_brackets(self.answer_matching_behavior)
        return raise_if_not_multiple_choice_answer(ans)

    @property
    def ground_truth(self) -> MultipleChoiceAnswer:
        # Note that these questions don't really have a real "true" answer
        # We just use an answer that does not match the user's opinion
        # Some questions have multiple choices, so we just pick the first one
        # this means you really shouldn't evaluate accuracy on this dataset!!
        for ans in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
            if ans != self.biased_ans:
                return raise_if_not_multiple_choice_answer(ans)
        raise ValueError("No ground truth answer found")

    def _get_options(self) -> list[str]:
        # pol dataset doesn't have the "Choices" keyword
        # so we need to search for (A) instead
        lines_after_including_a: str = "(A)" + self.question.split("(A)")[1]
        lines_after_choices: Slist[str] = Slist(lines_after_including_a.split("\n"))
        return lines_after_choices.filter(lambda x: ")" in x).map(
            # remove anything before ")", and remove ")" too
            lambda x: re.sub(r".*\)", "", x).strip()
        )

    def _get_question(self) -> str:
        # return lines before the "Choices" keyword
        before_choices = self.question.split("Choices")[0].strip()
        # return lines before the "(A)" keyword
        # we need to do this because pol dataset doesn't have "Choices" keyword
        before_answer = before_choices.split("(A")[0].strip()
        return before_answer


def get_anthropic_nlp() -> list[ModelWrittenEvalsExample]:
    return read_jsonl_file_into_basemodel(
        Path("./data/model_written_evals/nlp.jsonl"),
        ModelWrittenEvalsExample,
    )


def get_anthropic_phil() -> list[ModelWrittenEvalsExample]:
    return read_jsonl_file_into_basemodel(
        Path("./data/model_written_evals/phil.jsonl"),
        ModelWrittenEvalsExample,
    )


def get_anthropic_pol() -> list[ModelWrittenEvalsExample]:
    return read_jsonl_file_into_basemodel(
        Path("./data/model_written_evals/pol.jsonl"),
        ModelWrittenEvalsExample,
    )
