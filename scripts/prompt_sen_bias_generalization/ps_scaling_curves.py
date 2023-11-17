import asyncio
from enum import Enum
import math
from collections import Counter
from pathlib import Path
from typing import Sequence, Type

import fire
import pandas as pd
from grugstream import Observable
from matplotlib import pyplot as plt
from pydantic import BaseModel
from slist import Slist, Group
from tqdm import tqdm

from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.data_models.config import OpenaiInferenceConfig, config_from_default
from cot_transparency.data_models.data import COT_TESTING_TASKS
from cot_transparency.data_models.data import COT_TRAINING_TASKS
from cot_transparency.data_models.example_base import DummyDataExample
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.data_models.pd_utils import BaseExtractor, BasicExtractor, convert_slist_to_df
from cot_transparency.data_models.streaming import ParaphrasedQuestion, StreamingTaskSpec
from cot_transparency.data_models.streaming import ParaphrasedTaskSpec
from cot_transparency.data_models.streaming import ParaphrasingOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.interventions.few_shots_loading import ModelOutputVerified, get_training_cots_gpt_35
from cot_transparency.formatters.prompt_sensitivity.automated_generations import (
    GenerateParaphrasingsFormatter2,
    GenerateParaphrasingsFormatters,
    GenerateParaphrasingsNoCotFormatters,
)
from cot_transparency.json_utils.read_write import (
    read_jsonl_file_into_basemodel,
    read_jsonl_file_into_basemodel_ignore_errors,
    write_jsonl_file_from_basemodel,
)
from cot_transparency.data_models.streaming import StreamingTaskOutput
from cot_transparency.streaming.tasks import call_model_with_task_spec
from cot_transparency.streaming.tasks import data_to_task_spec
from cot_transparency.streaming.tasks import get_examples_for_tasks
from scripts.automated_answer_parsing.answer_parsing_example import answer_finding_step
from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    FormatterWithPossibleIntervention,
    ParaphrasingSampler,
)
from scripts.finetune_zero_shot_experiments.comparison_plot import FilterStrategy, ModelTrainMeta
from scripts.prompt_sen_bias_generalization.model_sweeps import SweepDatabase, Sweeps
from scripts.prompt_sen_bias_generalization.model_sweeps.paraphrasing import (
    BASELINE_1_W_VERBALIZE,
    BASELINE_1_W_VERBALIZE_CORRECT,
    PARAPHRASING_1_W_VERBALIZE,
    PARAPHRASING_1_W_VERBALIZE_CORRECT,
    PARAPHRASING_2_W_VERBALIZE,
    PARAPHRASING_2_W_VERBALIZE_CORRECT,
)
from scripts.prompt_sen_bias_generalization.util import lineplot_util, load_per_model_results, save_per_model_results
from scripts.prompt_sen_bias_generalization.util import add_point_at_1
from scripts.utils.plots import catplot

SWEEPS_DB = SweepDatabase()
SWEEPS_DB.add(Sweeps.gpt)
# SWEEPS_DB.add(Sweeps.zero_shot_2)
# SWEEPS_DB.add(Sweeps.few_shot_2)
# SWEEPS_DB.add(Sweeps.paraphrasing_2_correct)
# SWEEPS_DB.add(Sweeps.paraphrasing_2_ba)
SWEEPS_DB.add(PARAPHRASING_1_W_VERBALIZE)
SWEEPS_DB.add(PARAPHRASING_2_W_VERBALIZE)
SWEEPS_DB.add(BASELINE_1_W_VERBALIZE)
SWEEPS_DB.add(BASELINE_1_W_VERBALIZE_CORRECT)
SWEEPS_DB.add(PARAPHRASING_1_W_VERBALIZE_CORRECT)
SWEEPS_DB.add(PARAPHRASING_2_W_VERBALIZE_CORRECT)


# SWEEPS_DB.add(Sweeps.paraphrasing_2_correct)
# SWEEPS_DB.add(Sweeps.gs_unbiased)
# SWEEPS_DB.add(Sweeps.prompt_variants_2)

EXP_DIR = "experiments/automated_prompt_variant_generation"
EVAL_DIR = f"{EXP_DIR}/evaluation_v2"
TRAINING_DIR = f"{EXP_DIR}/training_data_v2"


def parse_responses(output: StreamingTaskOutput) -> ParaphrasingOutput:
    model_response = output.inference_output.raw_response
    outputs = GenerateParaphrasingsFormatters.get_paraphrased_questions(model_response)
    paraphrased_questions = Slist(outputs).map(lambda x: ParaphrasedQuestion(paraphrased=x[0], tags=x[1]))

    return ParaphrasingOutput(
        task_spec=output.task_spec,
        inference_output=output.inference_output,
        paraphrased_questions=paraphrased_questions,
    )


def reformulate_questions_for_asking(
    x: ParaphrasingOutput,
    configs: Sequence[OpenaiInferenceConfig],
    formatters: Sequence[Type[StageOneFormatter]] = [ZeroShotCOTUnbiasedFormatter],
) -> Sequence[ParaphrasedTaskSpec]:
    specs = []

    for paraphrased_question in x.paraphrased_questions:
        paraphrased_as_data_example = DummyDataExample(parsed_input=paraphrased_question.paraphrased)
        for config in configs:
            for f in formatters:
                messages = f.format_example(paraphrased_as_data_example, model=config.model)
                ts = ParaphrasedTaskSpec(
                    messages=messages,
                    formatter_name=f.name(),
                    data_example=x.task_spec.data_example,
                    inference_config=config,
                    task_name=x.task_spec.task_name,
                    paraphrased_question=paraphrased_question,
                )
                specs.append(ts)
    return specs


class CotTasks(str, Enum):
    training = "training"
    testing = "testing"


async def run_pipeline(
    exp_dir: str = EVAL_DIR,
    example_cap: int = 200,
    tasks: CotTasks = CotTasks.testing,
    batch_size: int = 50,
    eval_temp: float = 0.0,
    models_to_evaluate: Sequence[str] = SWEEPS_DB.all_model_names,
    paraphrasing_formatters: Sequence[Type[StageOneFormatter]] = [GenerateParaphrasingsFormatter2],
    # GenerateParahrasingsFormatter2 doesn't specify whether to use COT or not so we add that with an intervention
) -> Path:
    cache_dir = f"{exp_dir}/cache"

    generation_caller = UniversalCaller().with_file_cache(f"{cache_dir}/generation_cache.jsonl", write_every_n=1)

    match tasks:
        case CotTasks.training:
            task_list = COT_TRAINING_TASKS
        case CotTasks.testing:
            task_list = COT_TESTING_TASKS

    data_examples = get_examples_for_tasks(task_list, example_cap)
    task_specs = data_examples.map(
        lambda x: data_to_task_spec(
            *x,
            formatters=paraphrasing_formatters,
            models=[config_from_default(model="gpt-4", max_tokens=6000)],
        )
    ).flatten_list()

    pipeline = (
        Observable.from_iterable(task_specs)
        .map_blocking_par(
            # We retry to make sure we get 10 paraphrasings per question, sometimes gpt gives fewer
            lambda x: call_model_with_task_spec(x, generation_caller, num_tries=20, should_raise=True),
            max_par=batch_size,
        )
        .flatten_list()
        .tqdm(tqdm_bar=tqdm(total=len(task_specs), desc="Generating prompts"))
        .map(parse_responses)
        .for_each_to_file(Path(f"{exp_dir}/paraphrasing_outputs.jsonl"), serialize=lambda x: x.model_dump_json())
    )
    if models_to_evaluate:
        pipeline = make_evaluation_pipeline(
            pipeline,
            batch_size,
            eval_temp,
            models_to_evaluate,
            cache_dir,
            task_specs,
        )

    results_dir = Path(exp_dir) / "results"
    results = await pipeline.to_slist()
    save_per_model_results(results, results_dir)

    return results_dir


def make_evaluation_pipeline(
    input_pipeline: Observable[ParaphrasingOutput],
    batch_size: int,
    eval_temp: float,
    models_to_evaluate: Sequence[str],
    cache_dir: str | Path,
    task_specs: Sequence[StreamingTaskSpec],
) -> Observable[StreamingTaskOutput]:
    models_to_be_tested = Slist(models_to_evaluate).map(lambda x: config_from_default(model=x, temperature=eval_temp))
    testing_caller = UniversalCaller().with_model_specific_file_cache(
        f"{cache_dir}/evaluation_cache", write_every_n=200
    )
    answer_parsing_caller = UniversalCaller().with_model_specific_file_cache(
        f"{cache_dir}/answer_parsing_cache", write_every_n=200
    )
    answer_parsing_config = config_from_default(model="claude-2")

    pipeline = (
        input_pipeline.map(
            lambda x: reformulate_questions_for_asking(
                x,
                models_to_be_tested,
            )
        )
        .flatten_iterable()
        .map_blocking_par(lambda x: call_model_with_task_spec(x, testing_caller), max_par=batch_size)
        .tqdm(tqdm_bar=tqdm(total=10 * len(task_specs) * len(models_to_be_tested), desc="Asking parahrased questions"))
        .flatten_list()
        .map_blocking_par(
            lambda x: answer_finding_step(x, answer_parsing_caller, answer_parsing_config), max_par=batch_size
        )
        .tqdm(tqdm_bar=tqdm(total=10 * len(task_specs) * len(models_to_be_tested), desc="Parsing Answers"))
    )

    return pipeline


def make_training_data(
    exp_dir=TRAINING_DIR,
    example_cap: int = 100000,
    tasks: CotTasks = CotTasks.training,
    gold_standard_formatters: Sequence[Type[StageOneFormatter]] = [],
    batch_size=50,
    v1: bool = False,
    evaluate_on_training_data: bool = False,
):
    # paraphrasing_formatters: Sequence[Type[StageOneFormatter]] = [
    if v1:
        paraphrasing_formatters: Sequence[Type[StageOneFormatter]] = [
            GenerateParaphrasingsFormatters,
            GenerateParaphrasingsNoCotFormatters,
        ]
    else:
        paraphrasing_formatters: Sequence[Type[StageOneFormatter]] = [GenerateParaphrasingsFormatter2]

    # This generates the different paraphrasings of the questions
    results_path = asyncio.run(
        run_pipeline(
            exp_dir=exp_dir,
            example_cap=example_cap,
            tasks=tasks,
            batch_size=batch_size,
            models_to_evaluate=[],
            eval_temp=0.0,
            paraphrasing_formatters=paraphrasing_formatters,
        )
    )
    paraphrased_questions = load_per_model_results(results_path, ParaphrasingOutput)
    grouped = paraphrased_questions.group_by(lambda x: x.task_spec.formatter_name)
    # cot_para = Slist(paraphrased_questions).filter(lambda x: name_to_formatter(x.task_spec.formatter_name).is_cot)
    # non_cot_para = Slist(paraphrased_questions).filter(
    #     lambda x: not name_to_formatter(x.task_spec.formatter_name).is_cot
    # )
    grouped.for_each(
        lambda x: write_jsonl_file_from_basemodel(Path(f"data/training_paraphrasings/{x.key}.jsonl"), x.values)
    )
    print("saved paraphrasings")

    # but we also want to generate the gold standard completions that we will use to train the model
    # dont need to use any paraphrasings here

    async def get_gold_standard_cots(
        exp_dir: str,
        example_cap: int,
        tasks: Sequence[str],
        batch_size: int,
        num_completions_per_prompt: int = 1,
    ):
        model_caller = UniversalCaller().with_file_cache(
            f"{exp_dir}/cache/cot_generation_cache.jsonl", write_every_n=20
        )
        data_examples = get_examples_for_tasks(tasks, example_cap)
        pipeline = (
            Observable.from_iterable(data_examples)
            .map(
                lambda x: data_to_task_spec(
                    *x,
                    formatters=gold_standard_formatters,
                    models=[config_from_default(model="gpt-3.5-turbo", max_tokens=3000, n=num_completions_per_prompt)],
                )
            )
            .flatten_iterable()
            .map_blocking_par(lambda x: call_model_with_task_spec(x, model_caller), max_par=batch_size)
            .flatten_iterable()
            .tqdm(
                tqdm_bar=tqdm(
                    total=len(data_examples) * len(gold_standard_formatters),
                    desc="Generating Gold Standard Completions",
                )
            )
        )

        results_path = Path(f"{exp_dir}/gold_standard_completions.jsonl")
        # delete the file if it exists
        if results_path.exists():
            results_path.unlink()
        await pipeline.to_file(results_path, mode="a", serialize=lambda x: x.model_dump_json())

        outputs = read_jsonl_file_into_basemodel(results_path, StreamingTaskOutput)

        # Save the outputs to different files based on the formatter
        groupby = outputs.group_by(lambda x: x.task_spec.formatter_name)
        groupby.for_each(lambda x: write_jsonl_file_from_basemodel(Path(f"data/training_cots/{x.key}.jsonl"), x.values))

    if gold_standard_formatters:
        asyncio.run(
            get_gold_standard_cots(
                exp_dir=exp_dir,
                example_cap=example_cap,
                tasks=tasks,
                batch_size=batch_size,
            )
        )


class Entropy(BaseModel):
    entropy: float
    uniform_entropy: float
    task: str


def entropy_and_uniform_entropy(outputs: Sequence[StreamingTaskOutput]) -> Entropy:
    inference_outputs = Slist(outputs).map(lambda x: x.inference_output)
    parsed_responses = inference_outputs.map(lambda x: x.parsed_response)
    counts = Counter(parsed_responses)
    print(counts)
    probabilities = {k: v / len(outputs) for k, v in counts.items()}
    entropy = -sum([p * math.log(p, 2) for p in probabilities.values()])

    # also return the entropy as if the model was uniform
    # get the number of options from the question
    num_options = Slist(outputs).map(lambda x: x.task_spec.n_options_given)
    assert len(num_options.distinct()) == 1
    uniform_prob = 1 / num_options[0]
    uniform_entropy = -sum([uniform_prob * math.log(uniform_prob, 2) for _ in range(num_options[0])])
    return Entropy(entropy=entropy, uniform_entropy=uniform_entropy, task=outputs[0].task_spec.task_name)


grouped_outputs = Group[tuple[str, OpenaiInferenceConfig], Entropy]


class Extractor(BaseExtractor[grouped_outputs]):
    column_names: list[str] = [
        "task_hash",
        "model",
        "model_with_temp",
        "temperature",
        "entropy",
        "uniform_entropy",
        "task",
    ]

    def extract(self, output: grouped_outputs) -> Sequence[str | float | None | bool]:
        task_hash = output[0][0]
        model = output[0][1].model
        temperature = output[0][1].temperature
        entropy = output[1]
        return [
            task_hash,
            model,
            f"{model}, t={temperature}",
            temperature,
            entropy.entropy,
            entropy.uniform_entropy,
            entropy.task,
        ]


def plot(exp_dir="experiments/automated_prompt_variant_generation/evaluation_v2"):
    results_dir = f"{exp_dir}/results"
    outputs = load_per_model_results(Path(results_dir), StreamingTaskOutput)

    # print counts by task
    print("Counts by task")
    counts = Slist(outputs).group_by(lambda x: x.task_spec.task_name).map(lambda x: (x.key, len(x.values)))
    print(counts)

    # calculate the entropy
    with_entropy = outputs.group_by(lambda x: (x.task_spec.get_task_hash(), x.task_spec.inference_config)).map(
        lambda x: x.map_values(entropy_and_uniform_entropy)
    )

    df = convert_slist_to_df(with_entropy, [Extractor()])

    # df["is_none"] = df.parsed_response.isna()

    avg_entropy = df.uniform_entropy.mean()
    catplot(data=df, x="model_with_temp", y="entropy", add_line_at=avg_entropy, hue="task")
    # catplot(data=df, x="model_with_temp", y="is_none", add_line_at=avg_entropy)

    plt.show()


def plot_scaling_curves(
    exp_dir=EVAL_DIR,
    model_meta: Sequence[ModelTrainMeta] = SWEEPS_DB.all_models,
    force_rerun: bool = False,
):
    model_meta = Slist(model_meta)
    model_name_to_meta = model_meta.map(lambda x: (x.name, x)).to_dict()

    # filter out to thoose that are only trained on correct cots
    model_meta = model_meta.filter(lambda x: x.filter_strategy == FilterStrategy.no_filter or x.name == "gpt-3.5-turbo")

    models = Slist(model_meta).map(lambda x: x.name)

    results_dir = f"{exp_dir}/results"

    outputs = load_per_model_results(results_dir, StreamingTaskOutput, model_names=models)
    loaded_models = Slist(outputs).map(lambda x: x.task_spec.inference_config.model).distinct()

    rerun = False
    for model in models:
        if model not in loaded_models:
            rerun = True
            print(f"Didn't find all models requested in {results_dir}. Running evaluation again.")
            break

    if rerun or force_rerun:
        results_dir = asyncio.run(
            run_pipeline(
                exp_dir=exp_dir,
                models_to_evaluate=models,
                tasks=CotTasks.testing,
                batch_size=60,
                eval_temp=0.0,
            )
        )

    # print number of responses per model
    print("Number of responses per model")
    print(Slist(outputs).group_by(lambda x: x.task_spec.inference_config.model).map(lambda x: (x.key, len(x.values))))

    orig_df = convert_slist_to_df(outputs, [BasicExtractor()])

    orig_df["Trained on COTS from"] = orig_df.model.map(lambda x: model_name_to_meta[x].for_legend())
    orig_df["Samples"] = orig_df.model.map(lambda x: model_name_to_meta[x].trained_samples)
    orig_df = add_point_at_1(orig_df, "gpt-3.5-turbo")

    # drop any sets of "Trained on COTS from" that only have samples at 1
    orig_df: pd.DataFrame = orig_df.groupby("Trained on COTS from").filter(lambda x: x.Samples.max() > 1)  # type: ignore
    orig_df["Accuracy"] = orig_df["parsed_response"] == orig_df["ground_truth"]
    orig_df["IsNone"] = 1 * orig_df["parsed_response"].isna()

    lineplot_util(orig_df, title="Accuracy", y="Accuracy")
    lineplot_util(orig_df, title="Proportion of responses that are None", y="IsNone")

    with_entropy = outputs.group_by(lambda x: (x.task_spec.get_task_hash(), x.task_spec.inference_config)).map(
        lambda x: x.map_values(entropy_and_uniform_entropy)
    )
    df = convert_slist_to_df(with_entropy, [Extractor()])
    df["Trained on COTS from"] = df.model.map(lambda x: model_name_to_meta[x].for_legend())
    df["Samples"] = df.model.map(lambda x: model_name_to_meta[x].trained_samples)
    df = add_point_at_1(df, "gpt-3.5-turbo")

    # drop any sets of "Trained on COTS from" that only have samples at 1
    df: pd.DataFrame = df.groupby("Trained on COTS from").filter(lambda x: x.Samples.max() > 1)  # type: ignore

    avg_entropy = df.uniform_entropy.mean()
    lineplot_util(df, title="Entropy across paraphrased questions", add_line_at=avg_entropy, y="entropy")

    plt.show()


SWEEPS_DB2 = SweepDatabase()
SWEEPS_DB2.add(Sweeps.gpt)

model_meta = ModelTrainMeta(
    name="ft:gpt-3.5-turbo-0613:far-ai::8KKW9NRb",
    trained_samples=10000,
    filter_strategy=FilterStrategy.no_filter,
    train_formatters=FormatterOptions.ask_paraphrased,
    sampling_strategy=ParaphrasingSampler(1),
    data_from=DataFromOptions.gpt_35_turbo,
)
SWEEPS_DB2.add([model_meta])
gpt4 = ModelTrainMeta(
    name="gpt-4",
    trained_samples=1,
    filter_strategy=FilterStrategy.no_filter,
    train_formatters=FormatterOptions.ask_paraphrased,
    sampling_strategy=ParaphrasingSampler(1),
    data_from=DataFromOptions.gpt_35_turbo,
)

SWEEPS_DB2.add([gpt4])


async def evaluate_on_training_data(
    training_file_path: str | Path,
    model_meta: Sequence[ModelTrainMeta] = SWEEPS_DB2.all_models,
    exp_dir: str = "experiments/automated_prompt_variant_generation/evaluation_trainset",
    batch_size: int = 50,
) -> None:
    # cot_paraphrasings_file = "data/training_paraphrasings/GenerateParaphrasingsFormatter2.jsonl"
    # non_cot_paraphrasings_file = "data/training_paraphrasings/GenerateParaphrasingsFormatter2.jsonl"

    cache_dir = Path(exp_dir) / "cache"

    # verify that the samples returned by this are the same as the ones used in the training file
    loaded = read_jsonl_file_into_basemodel_ignore_errors(Path(training_file_path), FinetuneSample)
    print(len(loaded))
    # print(samples[0])
    cot_data = Slist(get_training_cots_gpt_35(ModelOutputVerified.unfiltered)).shuffle(str(42))

    # print counts by task
    print("Counts by task")
    counts = cot_data.group_by(lambda x: x.get_task_spec().task_name).map(lambda x: (x.key, len(x.values)))
    print(counts)

    loaded_responses = (
        Slist(loaded).map(lambda x: x.messages[-1].content).map(lambda x: x.replace("Let's think step by step: ", ""))
    )

    def match_response(x: TaskOutput, loaded_responses: Sequence[str]):
        if x.inference_output.raw_response in loaded_responses:
            return True
        return False

    cot_data_that_was_used = Slist(cot_data).filter(lambda x: match_response(x, loaded_responses))
    print("Counts by task filtered")
    counts = cot_data_that_was_used.group_by(lambda x: x.get_task_spec().task_name).map(
        lambda x: (x.key, len(x.values))
    )
    print(counts)

    formatter = FormatterWithPossibleIntervention(formatter=ZeroShotCOTUnbiasedFormatter)
    cot_paraphrasings_file = "./data/training_paraphrasings/GenerateParaphrasingsFormatters.jsonl"
    non_cot_paraphrasings_file = "./data/training_paraphrasings/GenerateParaphrasingsNoCotFormatters.jsonl"
    sampler = ParaphrasingSampler(
        n_formats_per_question=10,
        cot_paraphrasings_file=cot_paraphrasings_file,
        non_cot_paraphrasings_file=non_cot_paraphrasings_file,
    )
    paraphrased = sampler.sample(cot_data_that_was_used, formatters=[formatter], n=3000)

    # convert to task specs
    configs = Slist(model_meta).map(lambda x: config_from_default(model=x.name, max_tokens=3000, n=1))
    f = ZeroShotCOTUnbiasedFormatter
    specs: Slist[StreamingTaskSpec] = Slist()
    for x in paraphrased:
        for config in configs:
            ts = StreamingTaskSpec(
                messages=x.get_task_spec().messages,
                formatter_name=f.name(),
                data_example=x.get_task_spec().get_data_example_obj().model_dump(),
                inference_config=config,
                task_name=x.get_task_spec().get_task_name(),
            )
            specs.append(ts)

    testing_caller = UniversalCaller().with_model_specific_file_cache(
        f"{cache_dir}/evaluation_cache", write_every_n=200
    )
    answer_parsing_caller = UniversalCaller().with_model_specific_file_cache(
        f"{cache_dir}/answer_parsing_cache", write_every_n=200
    )
    answer_parsing_config = config_from_default(model="claude-2")

    pipeline = (
        Observable.from_iterable(specs)
        .map_blocking_par(lambda x: call_model_with_task_spec(x, testing_caller), max_par=batch_size)
        .tqdm(tqdm_bar=tqdm(total=len(specs), desc="Asking parahrased questions"))
        .flatten_list()
        .map_blocking_par(
            lambda x: answer_finding_step(x, answer_parsing_caller, answer_parsing_config), max_par=batch_size
        )
        .tqdm(tqdm_bar=tqdm(total=len(specs), desc="Parsing Answers"))
    )

    results = await pipeline.to_slist()

    results_dir = Path(exp_dir) / "results"
    save_per_model_results(results, results_dir)


if __name__ == "__main__":
    fire.Fire(
        {
            "run": run_pipeline,
            "plot": plot,
            "make_training_data": make_training_data,
            "plot_scaling_curves": plot_scaling_curves,
            "evaluate_on_training_data": evaluate_on_training_data,
        }
    )
