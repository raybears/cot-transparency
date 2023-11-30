from enum import Enum
from pathlib import Path
from typing import Sequence, Type
from matplotlib import pyplot as plt
from slist import Slist
import fire
from grugstream import Observable
from tqdm import tqdm

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.config import OpenaiInferenceConfig, config_from_default
from cot_transparency.data_models.data import COT_TRAINING_TASKS
from cot_transparency.data_models.example_base import DummyDataExample
from cot_transparency.data_models.models import BaseTaskOutput, TaskOutput
from cot_transparency.data_models.pd_utils import BaseExtractor, BasicExtractor, convert_slist_to_df
from cot_transparency.data_models.streaming import StreamingTaskSpec
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.more_biases.gsm import AskGSMQuestion, GSMAnswerFinder
from cot_transparency.formatters.prompt_sensitivity.automated_generations import (
    AddSpuriousInfoFormatter,
)
from cot_transparency.data_models.streaming import StreamingTaskOutput
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from cot_transparency.streaming.tasks import call_model_with_task_spec
from cot_transparency.streaming.tasks import data_to_task_spec
from cot_transparency.streaming.tasks import get_examples_for_tasks
from cot_transparency.util import assert_not_none
from scripts.automated_answer_parsing.answer_parsing_example import answer_finding_step
from scripts.prompt_sen_bias_generalization.util import load_per_model_results, save_per_model_results
from scripts.utils.plots import catplot


EXP_DIR = "experiments/s2a"
EVAL_DIR = f"{EXP_DIR}/evaluation"
GSM_DIR = f"{EXP_DIR}/gsm"

MODEL_NAME_MAP = {
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7p2hsv": "Intervention",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNFqGeq": "Paraphrasing",
    "ft:gpt-3.5-turbo-0613:far-ai::8NPtWM2y": "All Zero-Shot",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNz4qzi": "Paraphrasing\n+ All Zero-Shot",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ": "Control",
    "gpt-3.5-turbo-0613": "GPT-3.5-Turbo",
}


def reformulate_questions_for_asking(
    x: StreamingTaskOutput,
    configs: Sequence[OpenaiInferenceConfig],
    formatters: Sequence[Type[StageOneFormatter]] = [ZeroShotCOTUnbiasedFormatter],
) -> Sequence[StreamingTaskSpec]:
    specs = []
    paraphrased_as_data_example = DummyDataExample(parsed_input=assert_not_none(x.inference_output.parsed_response))
    for config in configs:
        for f in formatters:
            messages = f.format_example(paraphrased_as_data_example, model=config.model)
            ts = StreamingTaskSpec(
                messages=messages,
                formatter_name=f.name(),
                data_example=x.task_spec.data_example,
                inference_config=config,
                task_name=x.task_spec.task_name,
            )
            specs.append(ts)
    return specs


class CotTasks(str, Enum):
    training = "training"
    testing = "testing"


async def run_pipeline(
    exp_dir: str = EVAL_DIR,
    example_cap: int = 1000,
    tasks: CotTasks = CotTasks.testing,
    batch_size: int = 50,
    eval_temp: float = 0.0,
    models_to_evaluate: Sequence[str] = list(MODEL_NAME_MAP.keys()),
    paraphrasing_formatters: Sequence[Type[StageOneFormatter]] = [AddSpuriousInfoFormatter],
    # GenerateParahrasingsFormatter2 doesn't specify whether to use COT or not so we add that with an intervention
) -> Path:
    cache_dir = f"{exp_dir}/cache"

    generation_caller = UniversalCaller().with_file_cache(f"{cache_dir}/generation_cache.jsonl", write_every_n=1)

    answer_parsing_caller = UniversalCaller().with_model_specific_file_cache(
        f"{cache_dir}/answer_parsing_cache", write_every_n=2
    )
    answer_parsing_config = config_from_default(model="claude-2")

    match tasks:
        case CotTasks.training:
            task_list = COT_TRAINING_TASKS
        case CotTasks.testing:
            task_list = ["mmlu"]

    data_examples = get_examples_for_tasks(task_list, example_cap)
    task_specs = data_examples.map(
        lambda x: data_to_task_spec(
            *x,
            formatters=paraphrasing_formatters,
            models=[config_from_default(model="gpt-4", max_tokens=6000)],
        )
    ).flatten_list()

    models_to_be_tested = Slist(models_to_evaluate).map(lambda x: config_from_default(model=x, temperature=eval_temp))
    testing_caller = UniversalCaller().with_model_specific_file_cache(f"{cache_dir}/evaluation_cache", write_every_n=1)

    pipeline = (
        Observable.from_iterable(task_specs)
        .map_blocking_par(
            # We retry to make sure we get 10 paraphrasings per question, sometimes gpt gives fewer
            lambda x: call_model_with_task_spec(x, generation_caller, num_tries=20, should_raise=True),
            max_par=batch_size,
        )
        .flatten_list()
        .tqdm(tqdm_bar=tqdm(total=len(task_specs), desc="Generating prompts"))
        .map(lambda x: reformulate_questions_for_asking(x, models_to_be_tested))
        .flatten_iterable()
        .map_blocking_par(lambda x: call_model_with_task_spec(x, testing_caller), max_par=batch_size)
        .tqdm(tqdm_bar=tqdm(total=len(task_specs) * len(models_to_be_tested), desc="Asking parahrased questions"))
        .flatten_list()
        .map_blocking_par(
            lambda x: answer_finding_step(x, answer_parsing_caller, answer_parsing_config), max_par=batch_size
        )
        .tqdm(tqdm_bar=tqdm(total=len(task_specs) * len(models_to_be_tested), desc="Parsing Answers"))
    )

    results_dir = Path(exp_dir) / "results"
    results = await pipeline.to_slist()
    save_per_model_results(results, results_dir)

    return results_dir


async def run_gsm(
    exp_dir: str = GSM_DIR,
    example_cap: int = 1000,
    batch_size: int = 50,
    eval_temp: float = 1.0,
    models_to_evaluate: Sequence[str] = list(MODEL_NAME_MAP.keys()),
):
    formatter = AskGSMQuestion
    caller = UniversalCaller().with_file_cache(f"{exp_dir}/cache/eval_cache.jsonl", write_every_n=1)

    obs = stage_one_stream(
        tasks=["gsm_biased", "gsm_unbiased"],
        formatters=[formatter.name()],
        models=models_to_evaluate,
        exp_dir=exp_dir,
        example_cap=example_cap,
        batch=batch_size,
        temperature=eval_temp,
        caller=caller,
        num_tries=1,
        raise_after_retries=False,
        should_log_parsing_failures=False,
    )

    answer_parsing_caller = UniversalCaller().with_model_specific_file_cache(
        f"{exp_dir}/cache/answer_parsing_cache", write_every_n=10
    )
    answer_parsing_config = config_from_default(model="claude-2")

    obs = obs.map_blocking_par(
        lambda x: answer_finding_step(
            x, caller=answer_parsing_caller, config=answer_parsing_config, answer_finding_formatter=GSMAnswerFinder
        )
    )
    res = await obs.to_slist()
    save_per_model_results(res, exp_dir + "/results")


class RawGroundTruthExtractor(BaseExtractor[TaskOutput]):
    column_names = [
        "raw_ground_truth",
    ]

    def extract(self, output: BaseTaskOutput) -> Sequence[str]:
        return [output.get_task_spec().get_data_example_obj()._ground_truth]


def plot(exp_dir: str = GSM_DIR):
    results_dir = f"{exp_dir}/results"
    outputs = load_per_model_results(Path(results_dir), TaskOutput)

    df = convert_slist_to_df(outputs, [BasicExtractor(), RawGroundTruthExtractor()])
    df["accuracy"] = df["parsed_response"] == df["raw_ground_truth"]
    df["model"] = df["model"].map(lambda x: MODEL_NAME_MAP[x])

    # count number of Nones by each type
    print(df.groupby(["model", "task_name"]).apply(lambda x: (x.parsed_response == "None").sum()))

    # drop where they were none
    df = df[df.parsed_response != "None"]

    catplot(
        data=df,
        x="model",
        y="accuracy",
        hue="task_name",
        name_map={"accuracy": "% Accuracy", "model": "Model"},
        y_scale=100,
        font_scale=1.5,
    )

    plt.show()


if __name__ == "__main__":
    fire.Fire(
        {
            "run": run_pipeline,
            "run_gsm": run_gsm,
            "plot": plot,
        }
    )
