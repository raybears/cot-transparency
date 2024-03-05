from typing import Mapping, Sequence

from pydantic import BaseModel
from reactivex import Observable
from slist import Group, Slist
from zipp import Path
from cot_transparency.apis.openai.formatting import append_assistant_preferred_to_last_user
from cot_transparency.data_models.example_base import MultipleChoiceAnswer
from cot_transparency.data_models.messages import StrictChatMessage
from cot_transparency.data_models.models import TaskSpec
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.more_biases.anchor_initial_wrong import PostHocNoPlease
from cot_transparency.formatters.more_biases.distractor_fact import FirstLetterDistractor
from cot_transparency.formatters.more_biases.random_bias_formatter import RandomBiasedFormatter
from cot_transparency.formatters.more_biases.user_wrong_cot import DistractorAnswerWithoutInfluence, DistractorArgumentCorrectOrWrong, DistractorArgumentImportant, DistractorArgumentNoTruthfullyAnswer, DistractorArgumentNotsure, ImprovedDistractorArgument
from cot_transparency.formatters.more_biases.wrong_few_shot import WrongFewShotMoreClearlyLabelledAtBottom
from cot_transparency.formatters.verbalize.formatters import BlackSquareBiasedFormatter
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel

from stage_one import create_stage_one_task_specs




FORMATTERS_TO_DUMP = {
    RandomBiasedFormatter.name(): "suggested_answer",
    PostHocNoPlease.name(): "post_hoc",
    WrongFewShotMoreClearlyLabelledAtBottom.name(): "wrong_few_shot",
    BlackSquareBiasedFormatter.name(): "spurious_few_shot_squares",
    FirstLetterDistractor.name(): "distractor_fact",
    ImprovedDistractorArgument.name(): "distractor_argument",
    # DistractorAnswerWithoutInfluence.name(): "distractor_argument_2",
    # DistractorArgumentCorrectOrWrong.name(): "distractor_argument_3",
    # DistractorArgumentImportant.name(): "distractor_argument_4",
    # DistractorArgumentNotsure.name(): "distractor_argument_5",
    # DistractorArgumentNoTruthfullyAnswer.name(): "distractor_argument_6",
}

formatters_list = [formatter for formatter in FORMATTERS_TO_DUMP]

def rename_dataset_name(dataset_name: str) -> str:
    match dataset_name:
        case "truthful_qa":
            return "truthfulqa"
        case "mmlu_test":
            return "mmlu"
        case _:
            return dataset_name
"""
class StrictChatMessage(BaseModel):
    role: str
    content: str
"""

class StandardDatasetDump(BaseModel):
    original_question: str
    original_question_hash: str
    original_dataset: str
    unbiased_question: list[StrictChatMessage]
    biased_question: list[StrictChatMessage]
    bias_name: str
    ground_truth: MultipleChoiceAnswer

    @staticmethod
    def from_task_spec(task_spec: TaskSpec) -> "StandardDatasetDump":
        bias_messages = task_spec.messages
        assistant_on_user_side: list[StrictChatMessage] = append_assistant_preferred_to_last_user(bias_messages)

        unbiased_question: list[StrictChatMessage] = append_assistant_preferred_to_last_user(ZeroShotCOTUnbiasedFormatter.format_example(question=task_spec.get_data_example_obj()))
        biased_question = assistant_on_user_side
        bias_name: str = FORMATTERS_TO_DUMP[task_spec.formatter_name]
    
        return StandardDatasetDump(
            original_question=task_spec.get_data_example_obj().get_parsed_input(),
            original_question_hash=task_spec.task_hash,
            original_dataset=rename_dataset_name(task_spec.task_name),
            unbiased_question=unbiased_question,
            biased_question=biased_question,
            bias_name=bias_name,
            ground_truth=task_spec.ground_truth, # type: ignore
        )

    
        

def dump_data():
    # delete whole dataset_dumps folder if it exists
    tasks_to_run: Slist[TaskSpec] = Slist(create_stage_one_task_specs(
        dataset="cot_testing",
        models=["gpt-3.5-turbo-0613"],
        formatters=formatters_list,
        example_cap=2000, # 2000 each dataset in "mmlu", "truthful_qa", ""
        temperature=0,
        raise_after_retries=False,
        max_tokens=1000,
        n_responses_per_request=1,
    ))
    dumps = tasks_to_run.map(StandardDatasetDump.from_task_spec)
    
    # put in a folder titled "dataset_dumps/test". The file will be named "{original_dataset}_{bias_name}.jsonl"
    # make the folder if it doesn't exist
    
    # group by this name
    dumps_grouped: Slist[Group[str, Slist[StandardDatasetDump]]] = dumps.group_by(lambda x: f"{x.original_dataset}_{x.bias_name}.jsonl")
    for group in dumps_grouped:
        write_jsonl_file_from_basemodel(f"dataset_dumps/test/{group.key}", group.values)

def test_parse_one_file():
    # open dataset_dumps/test/mmlu_distractor_fact.jsonl
    with open("dataset_dumps/test/mmlu_distractor_fact.jsonl", "r") as f:
        for line in f.readlines():
            # read into the basemodel
            parsed = StandardDatasetDump.model_validate_json(line)
            print(parsed.biased_question)


if __name__ == "__main__":
    dump_data()
    test_parse_one_file()

