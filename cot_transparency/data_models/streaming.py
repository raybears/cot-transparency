from pydantic import BaseModel
from cot_transparency.data_models.models import ModelOutput
from typing import Sequence
from cot_transparency.streaming.tasks import StreamingTaskOutput, StreamingTaskSpec


class ParaphrasedQuestion(BaseModel):
    paraphrased: str
    tags: Sequence[str]


class ParaphrasedTaskSpec(StreamingTaskSpec):
    paraphrased_question: ParaphrasedQuestion


class ParaphrasingOutput(StreamingTaskOutput):
    task_spec: StreamingTaskSpec
    inference_output: ModelOutput
    paraphrased_questions: Sequence[ParaphrasedQuestion]