from typing import Optional, Type
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.interventions.intervention import Intervention


class StepByStep(Intervention):
    n_samples: int = 10

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
