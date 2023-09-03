from pydantic import BaseModel
from slist import Slist

from cot_transparency.data_models.models import ChatMessage, TaskOutput
from cot_transparency.formatters.interventions.assistant_completion_utils import (
    add_to_final_assistant,
    prepend_to_front_system_message,
)
from cot_transparency.formatters.instructions import END_SINGLE_SHOT_SEP, UNBIASED_CONTROL_TOKEN, BIASED_CONTROL_TOKEN
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

    def to_finetune_sample_control_tokens(self) -> Slist[FinetuneSample]:
        # For the biased response, add NN in the system prompt
        # For the unbiased response, add YY in the system prompt
        prompt_messages = self.biased_question
        added_system_unbiased: list[ChatMessage] = prepend_to_front_system_message(
            messages=prompt_messages, prepend=f"{UNBIASED_CONTROL_TOKEN} "
        )
        added_system_biased: list[ChatMessage] = prepend_to_front_system_message(
            messages=prompt_messages, prepend=f"{BIASED_CONTROL_TOKEN} "
        )
        strict_unbiased = Prompt(
            messages=join_assistant_preferred_to_completion(
                messages=added_system_unbiased, completion=self.correct_full_response
            )
        )
        strict_biased = Prompt(
            messages=join_assistant_preferred_to_completion(
                messages=added_system_biased, completion=self.incorrect_full_response
            )
        )
        return Slist(
            [
                FinetuneSample(messages=strict_unbiased.get_strict_messages(ModelType.chat)),
                FinetuneSample(messages=strict_biased.get_strict_messages(ModelType.chat)),
            ]
        )


def format_big_brain_question_cot(task: BiasedQuestionUnbiasedCOT) -> Prompt:
    biased_messages: list[ChatMessage] = task.biased_question
    with_correct = add_to_final_assistant(
        biased_messages,
        new_message=" " + task.correct_full_response + END_SINGLE_SHOT_SEP,
    )
    return Prompt(messages=with_correct)
