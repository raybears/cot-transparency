from typing import Optional, Type
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.instructions import COT_ASSISTANT_PROMPT, VERBALIZE_INSTRUCTION
from cot_transparency.formatters.interventions.intervention import Intervention


class StepByStep(Intervention):
    # Non cot, only the label
    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> list[ChatMessage]:
        questions = formatter.format_example(question, model=model)
        step_by_step = ChatMessage(role=MessageRole.assistant, content="Let's think step by step.")
        return questions + [step_by_step]


class AddStepByStepVariant(Intervention):
    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> list[ChatMessage]:
        questions = formatter.format_example(question, model=model)
        step_by_step = ChatMessage(role=MessageRole.assistant, content="mod")
        return questions + [step_by_step]


class AddVerbalizeInstructionBestAnswer(Intervention):
    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> list[ChatMessage]:
        questions = formatter.format_example(question, model=model)

        last_message = questions[-1]
        content = last_message.content + VERBALIZE_INSTRUCTION
        last_message = ChatMessage(role=last_message.role, content=content)
        questions[-1] = last_message
        return questions


class AddVerbalizeAndStepByStep(Intervention):
    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> list[ChatMessage]:
        questions = formatter.format_example(question, model=model)

        last_message = questions[-1]
        content = last_message.content + VERBALIZE_INSTRUCTION
        last_message = ChatMessage(role=last_message.role, content=content)
        questions[-1] = last_message
        questions.append(ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT))
        return questions


class AddVerbalizeInstruction(Intervention):
    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> list[ChatMessage]:
        questions = formatter.format_example(question, model=model)

        last_message = questions[-1]
        content = last_message.content + "\n\n" + VERBALIZE_INSTRUCTION
        last_message = ChatMessage(role=last_message.role, content=content)
        questions[-1] = last_message
        questions.append(ChatMessage(role=MessageRole.assistant, content="Let's think step by step."))
        return questions
