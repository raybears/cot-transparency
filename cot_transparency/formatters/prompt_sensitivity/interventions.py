"""
This file exists as the prompt variants are defined in quite a general way. They don't specify 
"please think step by step"  so if you want that behaviour then you can use one of these intervetions. 
This is used for consistency training.
"""
from typing import Optional, Sequence, Type

from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.instructions import (
    COT_ASSISTANT_PROMPT_TRAINING,
    NON_COT_ASSISTANT_PROMPT,
    VERBALIZE_INSTRUCTION,
    add_verbalize_instruction_to_question,
)
from cot_transparency.formatters.interventions.intervention import Intervention


class StepByStep(Intervention):
    # Non cot, only the label
    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        questions = list(formatter.format_example(question, model=model))
        step_by_step = ChatMessage(role=MessageRole.assistant, content="Let's think step by step.")
        return questions + [step_by_step]


class AddBestAnswerIsAssistant(Intervention):
    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        messages = list(formatter.format_example(question, model=model))
        messages.append(
            ChatMessage(
                role=MessageRole.assistant,
                content=NON_COT_ASSISTANT_PROMPT,
            )
        )
        return messages


class AddBestAnswerIsNonCot(Intervention):
    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        messages = list(formatter.format_example(question, model=model))
        messages.append(
            ChatMessage(
                role=MessageRole.assistant_if_completion,
                content=NON_COT_ASSISTANT_PROMPT,
            )
        )
        return messages


class AddVerbalizeInstructionBestAnswer(Intervention):
    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        questions = list(formatter.format_example(question, model=model))

        last_message = questions[-1]
        content = last_message.content + VERBALIZE_INSTRUCTION
        last_message = ChatMessage(role=last_message.role, content=content)
        questions[-1] = last_message
        return questions


class AddVerbalizeAndStepByStepAssistantPref(Intervention):
    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        questions = list(formatter.format_example(question, model=model))

        last_message = questions[-1]
        with_label_instruction = ChatMessage(
            role=last_message.role,
            content=add_verbalize_instruction_to_question(last_message.content),
        )
        questions[-1] = with_label_instruction
        questions.append(ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT_TRAINING))
        return questions


class AddVerbalizeInstruction(Intervention):
    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        questions = list(formatter.format_example(question, model=model))

        last_message = questions[-1]
        content = last_message.content + "\n\n" + VERBALIZE_INSTRUCTION
        last_message = ChatMessage(role=last_message.role, content=content)
        questions[-1] = last_message
        questions.append(ChatMessage(role=MessageRole.assistant, content="Let's think step by step."))
        return questions
