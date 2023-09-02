from pydantic import BaseModel

from cot_transparency.data_models.models import ChatMessage, TaskOutput
from cot_transparency.formatters.interventions.assistant_completion_utils import add_to_final_assistant
from cot_transparency.formatters.instructions import END_SINGLE_SHOT_SEP
from cot_transparency.model_apis import Prompt, ModelType
from cot_transparency.openai_utils.finetune import FinetuneSample, join_assistant_preferred_to_completion


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

    def to_prompt_with_unbiased_response(self) -> Prompt:
        return format_big_brain_question_cot(task=self)

    def to_finetune_sample(self) -> FinetuneSample:
        prompt_messages = self.biased_question
        joined = join_assistant_preferred_to_completion(messages=prompt_messages, completion=self.correct_full_response)
        strict = Prompt(messages=joined)
        return FinetuneSample(messages=strict.get_strict_messages(ModelType.chat))


def format_big_brain_question_cot(task: BiasedQuestionUnbiasedCOT) -> Prompt:
    biased_messages: list[ChatMessage] = task.biased_question
    with_correct = add_to_final_assistant(
        biased_messages,
        new_message=" " + task.correct_full_response + END_SINGLE_SHOT_SEP,
    )
    return Prompt(messages=with_correct)
