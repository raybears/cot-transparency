from pydantic import BaseModel

from cot_transparency.data_models.models import ChatMessage, TaskOutput


class BiasedQuestionUnbiasedCOT(BaseModel):
    biased_question: list[ChatMessage]
    # the COT is full_response
    correct_full_response: str
    correct_parsed_response: str
    incorrect_full_response: str
    incorrect_parsed_response: str
    original_biased_task: TaskOutput
    original_unbiased_task: TaskOutput

    @property
    def incorrect_formatter_name(self) -> str:
        return self.original_biased_task.task_spec.formatter_name
