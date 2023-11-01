import asyncio
from collections import Counter
import math
from pathlib import Path
from typing import Generic, Sequence, Type, TypeVar
import fire

from grugstream import Observable
from matplotlib import pyplot as plt
from pydantic import BaseModel
from slist import Slist, Group
from tqdm import tqdm

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.config import OpenaiInferenceConfig, config_from_default
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.pd_utils import BaseExtractor, convert_slist_to_df
from cot_transparency.data_models.streaming import ParaphrasedTaskSpec
from cot_transparency.data_models.streaming import ParaphrasingOutput
from cot_transparency.data_models.streaming import ParaphrasedQuestion
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.interventions.few_shots_loading import ModelOutputVerified
from cot_transparency.formatters.prompt_sensitivity.automated_generations import (
    AskParaphrasedQuestionFormatter,
    GenerateParaphrasingsFormatters,
    GenerateParaphrasingsNoCotFormatters,
    GoldStandardNoCotFormatter,
    GoldStandardWithCotFormatter,
)
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel, write_jsonl_file_from_basemodel
from cot_transparency.streaming.tasks import StreamingTaskOutput
from cot_transparency.streaming.tasks import data_to_task_spec
from cot_transparency.streaming.tasks import call_model_with_task_spec
from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    NFormatsPerQuestionSampler,
    ParaphrasingSampler,
    fine_tune_with_bias_augmentation,
)
from scripts.utils.plots import catplot
from stage_one import COT_TESTING_TASKS, COT_TRAINING_TASKS, get_list_of_examples
from scripts.automated_answer_parsing.answer_parsing_example import answer_finding_step


def get_examples_for_tasks(tasks: Sequence[str], example_cap: int) -> Slist[tuple[str, DataExampleBase]]:
    """
    Returns a list of tuples of (task_name, example_obj)
    """
    ret = Slist()
    for t in tasks:
        examples = get_list_of_examples(t)
        # print(f"Found {len(examples)} examples for task: {t}")
        task_with_name = examples.map(lambda x: (t, x)).shuffle(str(42)).take(example_cap)
        ret.extend(task_with_name)
    return ret


def parse_responses(output: StreamingTaskOutput) -> ParaphrasingOutput:
    model_response = output.inference_output.raw_response
    outputs = GenerateParaphrasingsFormatters.get_paraphrased_questions(model_response)
    paraphrased_questions = Slist(outputs).map(lambda x: ParaphrasedQuestion(paraphrased=x[0], tags=x[1]))

    return ParaphrasingOutput(
        task_spec=output.task_spec,
        inference_output=output.inference_output,
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
    eval_temp: float,
    models_to_evaluate: Sequence[str] = [],
    paraphrasing_formatters: Sequence[Type[StageOneFormatter]] = [GenerateParaphrasingsFormatters],
):
    cache_dir = f"{exp_dir}/cache"

    generation_caller = UniversalCaller().with_file_cache(f"{cache_dir}/generation_cache.jsonl", write_every_n=1)

    answer_parsing_caller = UniversalCaller().with_model_specific_file_cache(f"{cache_dir}/answer_parsing_cache")
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
        .map_blocking_par(lambda x: call_model_with_task_spec(x, generation_caller), max_par=batch_size)
        .flatten_list()
        .tqdm(tqdm_bar=tqdm(total=n_items * len(paraphrasing_formatters), desc="Generating prompts"))
        .map(parse_responses)
    )
    if models_to_evaluate:
        models_to_be_tested = Slist(models_to_evaluate).map(
            lambda x: config_from_default(model=x, temperature=eval_temp)
        )
        testing_caller = UniversalCaller().with_model_specific_file_cache(f"{cache_dir}/evaluation_cache")

        pipeline = (
            pipeline.map(lambda x: reformulate_questions_for_asking(x, models_to_be_tested))
            .flatten_iterable()
            .map_blocking_par(lambda x: call_model_with_task_spec(x, testing_caller), max_par=batch_size)
            .tqdm(tqdm_bar=tqdm(total=n_items * 10 * len(models_to_evaluate), desc="Asking parahrased questions"))
            .flatten_list()
            .map_blocking_par(
                lambda x: answer_finding_step(x, answer_parsing_caller, answer_parsing_config), max_par=10
            )
            .tqdm(tqdm_bar=tqdm(total=n_items * 10 * len(models_to_evaluate), desc="Evaluating models"))
        )

    results_path = Path(f"{exp_dir}/results.jsonl")
    # delete the file if it exists
    if results_path.exists():
        results_path.unlink()
    await pipeline.to_file(results_path, mode="a", serialize=lambda x: x.model_dump_json())
    print("Done ✅")
    return results_path


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
    # This generates the different paraphrasings of the questions
    results_path = asyncio.run(
        run_pipeline(
            exp_dir=exp_dir,
            example_cap=example_cap,
            tasks=tasks,
            batch_size=batch_size,
            eval_temp=0.0,
            paraphrasing_formatters=paraphrasing_formatters,
        )
    )
    paraphrased_questions = read_jsonl_file_into_basemodel(results_path, ParaphrasingOutput)
    write_jsonl_file_from_basemodel(Path("data/training_paraphrasings/gpt4_paraphrasings.jsonl"), paraphrased_questions)

    # but we also want to generate the gold standard completions that we will use to train the model
    # dont need to use any paraphrasings here

    async def get_gold_standard_cots(
        exp_dir: str,
        example_cap: int,
        tasks: Sequence[str],
        batch_size: int,
    ):
        model_caller = UniversalCaller().with_file_cache(f"{exp_dir}/cache/cot_generation_cache.jsonl", write_every_n=1)
        data_examples = get_examples_for_tasks(tasks, example_cap)
        pipeline = (
            Observable.from_iterable(data_examples)
            .map(
                lambda x: data_to_task_spec(
                    *x,
                    formatters=[GoldStandardWithCotFormatter, GoldStandardNoCotFormatter],
                    models=[config_from_default(model="gpt-3.5-turbo", max_tokens=3000)],
                )
            )
            .flatten_iterable()
            .map_blocking_par(lambda x: call_model_with_task_spec(x, model_caller), max_par=batch_size)
            .flatten_iterable()
            .tqdm(tqdm_bar=tqdm(total=len(data_examples) * 2, desc="Generating Gold Standard Completions"))
        )

        results_path = Path(f"{exp_dir}/gold_standard_completions.jsonl")
        # delete the file if it exists
        if results_path.exists():
            results_path.unlink()
        await pipeline.to_file(results_path, mode="a", serialize=lambda x: x.model_dump_json())

        outputs = read_jsonl_file_into_basemodel(results_path, StreamingTaskOutput)
        cots = outputs.filter(lambda x: x.task_spec.formatter_name == GoldStandardWithCotFormatter.name())
        write_jsonl_file_from_basemodel(Path("data/training_cots/gpt35_gold_standard_cots.jsonl"), cots)
        non_cots = outputs.filter(lambda x: x.task_spec.formatter_name == GoldStandardNoCotFormatter.name())
        write_jsonl_file_from_basemodel(Path("data/training_non_cots/gpt35_gold_standard_cots.jsonl"), non_cots)

    asyncio.run(
        get_gold_standard_cots(
            exp_dir=exp_dir,
            example_cap=example_cap,
            tasks=tasks,
            batch_size=batch_size,
        )
    )


EXP_DIR = "experiments/automated_prompt_variant_generation/v1"


def run(
    exp_dir=EXP_DIR,
    models: Sequence[str] = [
        "gpt-3.5-turbo",
        "ft:gpt-3.5-turbo-0613:far-ai::8DPAu94W",  # super dataset 100k
        "ft:gpt-3.5-turbo-0613:far-ai::8Czg32py",  # super dataset 50k
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8CxBtbeH",  # super dataset 10k
    ],
    example_cap: int = 200,
    tasks: Sequence[str] = COT_TESTING_TASKS,
    batch_size: int = 50,
    eval_temp: float = 0.0,
):
    asyncio.run(
        run_pipeline(
            exp_dir=exp_dir,
            models_to_evaluate=models,
            batch_size=batch_size,
            eval_temp=eval_temp,
            example_cap=example_cap,
            tasks=tasks,
        )
    )


class Entropy(BaseModel):
    entropy: float
    uniform_entropy: float


def entropy_and_uniform_entropy(outputs: Slist[StreamingTaskOutput]) -> Entropy:
    inference_outputs = outputs.map(lambda x: x.inference_output)
    parsed_responses = inference_outputs.map(lambda x: x.parsed_response)
    counts = Counter(parsed_responses)
    print(counts)
    probabilities = {k: v / len(outputs) for k, v in counts.items()}
    entropy = -sum([p * math.log(p, 2) for p in probabilities.values()])

    # also return the entropy as if the model was uniform
    # get the number of options from the question
    num_options = outputs.map(lambda x: x.task_spec.n_options_given)
    assert len(num_options.distinct()) == 1
    uniform_prob = 1 / num_options[0]
    uniform_entropy = -sum([uniform_prob * math.log(uniform_prob, 2) for _ in range(num_options[0])])
    return Entropy(entropy=entropy, uniform_entropy=uniform_entropy)


grouped_outputs = tuple[Group[str, OpenaiInferenceConfig], Entropy]


class Extractor(BaseExtractor[grouped_outputs]):
    column_names: list[str] = ["task_hash", "model", "model_with_temp", "temperature", "entropy", "uniform_entropy"]

    def extract(self, output: grouped_outputs) -> Sequence[str | float | None | bool]:
        task_hash = output[0][0]
        model = output[0][1].model
        temperature = output[0][1].temperature
        entropy = output[1]
        return [task_hash, model, f"{model}, t={temperature}", temperature, entropy.entropy, entropy.uniform_entropy]


def plot(exp_dir="experiments/automated_prompt_variant_generation/v1"):
    experiment_path = f"{exp_dir}/results.jsonl"
    outputs = read_jsonl_file_into_basemodel(experiment_path, StreamingTaskOutput)

    # calculate the entropy
    with_entropy = outputs.group_by(lambda x: (x.task_spec.get_task_hash, x.task_spec.inference_config)).map(
        lambda x: x.map_values(entropy_and_uniform_entropy)
    )

    df = convert_slist_to_df(with_entropy, [Extractor()])

    avg_entropy = df.uniform_entropy.mean()
    catplot(data=df, x="model_with_temp", y="entropy", add_line_at=avg_entropy)

    plt.show()


def train_and_run(n_samples: int = 10000, n_formats_per_question: int = 2, unbiased: bool = False):
    if unbiased:
        assert n_formats_per_question == 1, "Only makes sense to have one format per question for unbiased"
        sampler = NFormatsPerQuestionSampler(n_formats_per_question=1)
        formatter_options = FormatterOptions.gs_unbiased
    else:
        sampler = ParaphrasingSampler(n_formats_per_question=n_formats_per_question)
        formatter_options = FormatterOptions.ask_paraphrased

    model = fine_tune_with_bias_augmentation(
        model="gpt-3.5-turbo",
        n_epochs=1,
        n_samples=n_samples,
        post_hoc=False,
        cot_percentage=0.50,
        project_name="consistency-training",
        formatter_options=formatter_options,
        sampler=sampler,
        permute_verbalize_instructions=False,
        data_from_options=DataFromOptions.gpt_35_turbo_gs,
        model_output_verified=ModelOutputVerified.unfiltered,
    )
    run(
        models=[model],
        tasks=COT_TESTING_TASKS,
        batch_size=50,
        eval_temp=0.0,
    )


if __name__ == "__main__":
    fire.Fire({"run": run, "plot": plot, "make_training_data": make_training_data, "train_and_run": train_and_run})
