import asyncio
from pathlib import Path
from typing import Generic, Sequence, Type, TypeVar
import fire

from grugstream import Observable
from pydantic import BaseModel
from slist import Slist
from tqdm import tqdm

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.config import OpenaiInferenceConfig, config_from_default
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import ModelOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.prompt_sensitivity.automated_generations import (
    AskParaphrasedQuestionFormatter,
    GenerateParaphrasingsFormatters,
    GenerateParaphrasingsNoCotFormatters,
)
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel
from cot_transparency.streaming import StreamingTaskOutput, StreamingTaskSpec
from cot_transparency.streaming import data_to_task_spec
from cot_transparency.streaming import model_step
from stage_one import COT_TESTING_TASKS, COT_TRAINING_TASKS, get_list_of_examples
from scripts.automated_answer_parsing.answer_parsing_example import answer_finding_step


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
        print(f"Found {len(examples)} examples for task: {t}")
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


async def run_pipeline(
    exp_dir: str,
    example_cap: int,
    tasks: Sequence[str],
    batch_size: int,
    models_to_evaluate: Sequence[str] = [],
    paraphrasing_formatters: Sequence[Type[StageOneFormatter]] = [GenerateParaphrasingsFormatters],
):
    generation_caller = UniversalCaller().with_file_cache(f"{exp_dir}/cache/generation_cache.jsonl", write_every_n=1)

    inter_file = PassThroughWriter[StreamingTaskOutput](f"{exp_dir}/generated_prompts.jsonl")
    paraphrased_file = PassThroughWriter[ParaphrasingOutput](f"{exp_dir}/paraphrased_prompts.jsonl")

    answer_parsing_caller = UniversalCaller().with_file_cache(f"{exp_dir}/cache/answer_parsing_cache.jsonl")
    answer_parsing_config = config_from_default(model="claude-v1")

    data_examples = get_examples_for_tasks(tasks, example_cap)
    n_items = len(data_examples)

    pipeline = (
        Observable.from_iterable(data_examples)
        .map(
            lambda x: data_to_task_spec(
                *x,
                formatters=paraphrasing_formatters,
                models=[config_from_default(model="gpt-4", max_tokens=3000)],
            )
        )
        .flatten_iterable()
        .map_blocking_par(lambda x: model_step(x, generation_caller), max_par=batch_size)
        .tqdm(tqdm_bar=tqdm(total=n_items, desc="Generating prompts"))
        .map(inter_file.write)
        .map(parse_responses)
        .map(paraphrased_file.write)
    )
    if len(models_to_evaluate):
        models_to_be_tested = Slist(models_to_evaluate).map(lambda x: config_from_default(model=x))
        testing_caller = UniversalCaller().with_file_cache(f"{exp_dir}/cache/evaluation_cache.jsonl")

        pipeline = (
            pipeline.map(lambda x: reformulate_questions_for_asking(x, models_to_be_tested))
            .flatten_iterable()
            .map_blocking_par(lambda x: model_step(x, testing_caller), max_par=batch_size)
            .map_blocking_par(
                lambda x: answer_finding_step(x, answer_parsing_caller, answer_parsing_config), max_par=10
            )
        )

    results_path = Path(f"{exp_dir}/results.jsonl")
    # delete the file if it exists
    if results_path.exists():
        results_path.unlink()
    await pipeline.to_file(results_path, mode="a", serialize=lambda x: x.model_dump_json())
    print("Done ✅")


def make_training_data(
    exp_dir="experiments/automated_prompt_variant_generation/training_data",
    example_cap: int = 2500,
    tasks: Sequence[str] = COT_TRAINING_TASKS,
    paraphrasing_formatters: Sequence[Type[StageOneFormatter]] = [
        GenerateParaphrasingsFormatters,
        GenerateParaphrasingsNoCotFormatters,
    ],
    batch_size=50,
):
    asyncio.run(
        run_pipeline(
            exp_dir=exp_dir,
            example_cap=example_cap,
            tasks=tasks,
            batch_size=batch_size,
            paraphrasing_formatters=paraphrasing_formatters,
        )
    )


def run(
    exp_dir="experiments/automated_prompt_variant_generation/v1",
    models: Sequence[str] = ["gpt-3.5-turbo"],
    example_cap: int = 200,
    tasks: Sequence[str] = COT_TESTING_TASKS,
    batch_size: int = 50,
):
    asyncio.run(
        run_pipeline(
            exp_dir=exp_dir,
            models_to_evaluate=models,
            example_cap=example_cap,
            tasks=tasks,
            batch_size=batch_size,
        )
    )


def plot():
    experiment_path = "/Users/edwardr/exp/cot/automated_prompt_variant_generation/v1/results.jsonl"
    read_jsonl_file_into_basemodel(experiment_path, StreamingTaskOutput)


if __name__ == "__main__":
    fire.Fire({"run": run, "plot": plot, "make_training_data": make_training_data})
