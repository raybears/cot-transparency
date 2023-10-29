import asyncio
from pathlib import Path
from typing import Generic, Sequence, TypeVar
import fire

from grugstream import Observable
from pydantic import BaseModel
from slist import Slist
from tqdm import tqdm

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.config import OpenaiInferenceConfig, config_from_default
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import ModelOutput
from cot_transparency.formatters.prompt_sensitivity.automated_generations import (
    AskParaphrasedQuestionFormatter,
    GenerateParaphrasingsFormatters,
)
from cot_transparency.streaming import StreamingTaskOutput, StreamingTaskSpec
from cot_transparency.streaming import data_to_task_spec
from cot_transparency.streaming import model_step
from stage_one import COT_TESTING_TASKS, get_list_of_examples


class ParaphrasedQuestion(BaseModel):
    paraphrased: str
    tags: Sequence[str]


class ParaphrasingOutput(StreamingTaskOutput):
    task_spec: StreamingTaskSpec
    inference_outputs: Sequence[ModelOutput]
    paraphrased_questions: Sequence[ParaphrasedQuestion]


class ParaphrasedTaskSpec(StreamingTaskSpec):
    paraphrased_question: ParaphrasedQuestion


def get_examples_for_tasks(tasks: Sequence[str], example_cap: int) -> Slist[tuple[str, DataExampleBase]]:
    """
    Returns a list of tuples of (task_name, example_obj)
    """
    ret = Slist()
    for t in tasks:
        examples = get_list_of_examples(t)
        task_with_name = examples.map(lambda x: (t, x)).shuffle(str(42)).take(example_cap)
        ret.extend(task_with_name)
    return ret


def parse_responses(output: StreamingTaskOutput) -> ParaphrasingOutput:
    model_responses = Slist(output.inference_outputs).map(lambda x: x.raw_response)
    outputs = model_responses.map(GenerateParaphrasingsFormatters.get_paraphrased_questions).flatten_list()
    paraphrased_questions = Slist(outputs).map(lambda x: ParaphrasedQuestion(paraphrased=x[0], tags=x[1]))

    return ParaphrasingOutput(
        task_spec=output.task_spec,
        inference_outputs=output.inference_outputs,
        paraphrased_questions=paraphrased_questions,
    )


A = TypeVar("A", bound=BaseModel)


class PassThroughWriter(Generic[A]):
    def __init__(self, path: str | Path):
        if isinstance(path, str):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "w")

    def write(self, x: A) -> A:
        self.fh.write(x.model_dump_json() + "\n")
        return x

    def __del__(self):
        self.fh.close()


def reformulate_questions_for_asking(
    x: ParaphrasingOutput, configs: Sequence[OpenaiInferenceConfig]
) -> Sequence[ParaphrasedTaskSpec]:
    specs = []
    for paraphrased_question in x.paraphrased_questions:
        for config in configs:
            messages = AskParaphrasedQuestionFormatter.format_example(
                paraphrased_question=paraphrased_question.paraphrased
            )
            ts = ParaphrasedTaskSpec(
                messages=messages,
                formatter_name=AskParaphrasedQuestionFormatter.name(),
                data_example=x.task_spec.data_example,
                inference_config=config,
                task_name=x.task_spec.task_name,
                paraphrased_question=paraphrased_question,
            )
            specs.append(ts)
    return specs


async def main(
    exp_dir="experiments/automated_prompt_variant_generation/v1",
    models: Sequence[str] = ["gpt-3.5-turbo"],
    example_cap: int = 2,
    tasks: Sequence[str] = COT_TESTING_TASKS,
    batch_size: int = 50,
):
    generation_caller = UniversalCaller().with_file_cache(f"{exp_dir}/cache/generation_cache.jsonl", write_every_n=1)

    inter_file = PassThroughWriter[StreamingTaskOutput](f"{exp_dir}/generated_prompts.jsonl")
    paraphrased_file = PassThroughWriter[ParaphrasingOutput](f"{exp_dir}/paraphrased_prompts.jsonl")

    models_to_be_tested = Slist(models).map(lambda x: config_from_default(model=x))
    testing_caller = UniversalCaller().with_file_cache(f"{exp_dir}/cache/evaluation_cache.jsonl")

    data_examples = get_examples_for_tasks(tasks, example_cap)
    n_items = len(data_examples)

    pipeline = (
        Observable.from_iterable(data_examples)
        .map(
            lambda x: data_to_task_spec(
                *x,
                formatters=[GenerateParaphrasingsFormatters],
                models=[config_from_default(model="gpt-4", max_tokens=3000)],
            )
        )
        .flatten_iterable()
        .map_blocking_par(lambda x: model_step(x, generation_caller), max_par=batch_size)
        .tqdm(tqdm_bar=tqdm(total=n_items, desc="Generating prompts"))
        .map(inter_file.write)
        .map(parse_responses)
        .map(paraphrased_file.write)
        .map(lambda x: reformulate_questions_for_asking(x, models_to_be_tested))
        .flatten_iterable()
        .map_blocking_par(lambda x: model_step(x, testing_caller), max_par=batch_size)
    )

    results_path = Path(f"{exp_dir}/results.jsonl")
    await pipeline.to_file(results_path, mode="w", serialize=lambda x: x.model_dump_json())
    print("Done ✅")


if __name__ == "__main__":
    asyncio.run(fire.Fire(main))
