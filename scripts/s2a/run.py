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
from cot_transparency.data_models.data import COT_TESTING_TASKS, COT_TRAINING_TASKS
from cot_transparency.data_models.example_base import DummyDataExample
from cot_transparency.data_models.models import BaseTaskOutput, TaskOutput
from cot_transparency.data_models.pd_utils import BaseExtractor, BasicExtractor, convert_slist_to_df
from cot_transparency.data_models.streaming import StreamingTaskSpec
from cot_transparency.formatters.prompt_sensitivity.automated_generations import (
    SPURIOUS_INFO_PROMPTS_CLEARLY_SPURIOUS,
    AddSpuriousInfoFormatterClearlySpurious,
)
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.more_biases.gsm import AskGSMQuestion, GSMAnswerFinder
from cot_transparency.data_models.streaming import StreamingTaskOutput
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
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
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNFqGeq": "Paraphrasing",
    # "ft:gpt-3.5-turbo-0613:far-ai::8NPtWM2y": "All Zero-Shot",
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNz4qzi": "Paraphrasing\n+ All Zero-Shot",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ": "Control",
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8QRaqhBm": "Middle",
    # "ft:gpt-3.5-turbo-0613:far-ai::8QdJtq3b": "Middle, All Zero-Shot split .",
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Qbpgb0B": "Middle, RBF split on .",
    "gpt-3.5-turbo-0613": "GPT-3.5-Turbo",
}


def reformulate_questions_for_asking(
    x: StreamingTaskOutput,
    configs: Sequence[OpenaiInferenceConfig],
    formatters: Sequence[Type[StageOneFormatter]] = [ZeroShotCOTUnbiasedFormatter],
    include_original: bool = True,
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
                paraphrasing_formatter_name=x.get_task_spec().formatter_name,
            )
            specs.append(ts)
            if include_original:
                ts = x.get_task_spec()
                messages = f.format_example(ts.get_data_example_obj(), model=config.model)
                ts = ts.copy_update(inference_config=config, messages=messages, formatter_name=f.name())
                specs.append(ts)

    return specs


class CotTasks(str, Enum):
    training = "training"
    testing = "testing"
    mmlu = "good_stuff"
    mmlu_test = "mmlu_test"
    testing_plus_aqua = "testing_plus_aqua"


async def run_pipeline(
    exp_dir: str = EVAL_DIR,
    example_cap: int = 2000,
    tasks: CotTasks = CotTasks.testing_plus_aqua,
    batch_size: int = 50,
    eval_temp: float = 0.0,
    models_to_evaluate: Sequence[str] = list(MODEL_NAME_MAP.keys()),
    paraphrasing_formatters: Sequence[Type[StageOneFormatter]] = [AddSpuriousInfoFormatterClearlySpurious],
    # GenerateParahrasingsFormatter2 doesn't specify whether to use COT or not so we add that with an intervention
) -> Path:
    cache_dir = f"{exp_dir}/cache"

    generation_caller = UniversalCaller().with_file_cache(f"{cache_dir}/generation_cache.jsonl", write_every_n=1)

    UniversalCaller().with_model_specific_file_cache(f"{cache_dir}/answer_parsing_cache", write_every_n=2)
    config_from_default(model="gpt-4")
    gen_output_path = SPURIOUS_INFO_PROMPTS_CLEARLY_SPURIOUS
    # Generation output:
    if gen_output_path.exists():
        # clear and start again
        gen_output_path.unlink()

    match tasks:
        case CotTasks.training:
            task_list = COT_TRAINING_TASKS
        case CotTasks.testing:
            task_list = COT_TESTING_TASKS
        case CotTasks.mmlu:
            task_list = ["mmlu"]
        case CotTasks.mmlu_test:
            task_list = ["mmlu_test"]
        case CotTasks.testing_plus_aqua:
            task_list = COT_TESTING_TASKS + ["aqua_train"]
    data_examples = get_examples_for_tasks(task_list, example_cap)

    task_specs = data_examples.map(
        lambda x: data_to_task_spec(
            *x,
            formatters=paraphrasing_formatters,
            models=[config_from_default(model="gpt-4", max_tokens=6000)],
        )
    ).flatten_list()

    Slist(models_to_evaluate).map(lambda x: config_from_default(model=x, temperature=eval_temp))
    UniversalCaller().with_model_specific_file_cache(f"{cache_dir}/evaluation_cache", write_every_n=1)

    pipeline = (
        Observable.from_iterable(task_specs)
        .map_blocking_par(
            lambda x: call_model_with_task_spec(x, generation_caller, num_tries=20, should_raise=False),
            max_par=batch_size,
        )
        .flatten_list()
        .for_each_to_file(gen_output_path, serialize=lambda x: x.model_dump_json())
        .tqdm(tqdm_bar=tqdm(total=len(task_specs), desc="Generating prompts"))
        # .map(lambda x: reformulate_questions_for_asking(x, models_to_be_tested))
        # .flatten_iterable()
        # .map_blocking_par(lambda x: call_model_with_task_spec(x, testing_caller), max_par=batch_size)
        # .tqdm(tqdm_bar=tqdm(total=len(task_specs) * len(models_to_be_tested), desc="Asking parahrased questions"))
        #     .flatten_list()
        #     .map_blocking_par(
        #         lambda x: answer_finding_step(x, answer_parsing_caller, answer_parsing_config), max_par=batch_size
        #     )
        #     .tqdm(tqdm_bar=tqdm(total=len(task_specs) * len(models_to_be_tested), desc="Parsing Answers"))
    )
    await pipeline.run_to_completion()

    results_dir = Path(exp_dir) / "results"
    results = await pipeline.to_slist()
    # save_per_model_results(results, results_dir)
    write_jsonl_file_from_basemodel(results_dir / "results.jsonl", results)

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
    write_jsonl_file_from_basemodel(exp_dir + "/results.jsonl", res)


class RawGroundTruthExtractor(BaseExtractor[BaseTaskOutput]):
    column_names = ["raw_ground_truth", "paraphrasing_formatter_name", "biased_ans", "label"]

    def extract(self, output: BaseTaskOutput) -> Sequence[str]:
        if isinstance(output, StreamingTaskOutput):
            para_formatter_name = output.get_task_spec().paraphrasing_formatter_name
            if para_formatter_name is None:
                para_formatter_name = "Original"
            label = "none"

        else:
            para_formatter_name = "Original"
            label = "blank label"

        return [
            output.get_task_spec().get_data_example_obj()._ground_truth,
            para_formatter_name,
            output.get_task_spec().get_data_example_obj().biased_ans,
            label,
        ]


def plot(
    exp_dir: str = GSM_DIR,
    hue: str = "task_name",
    filter_bias_on_incorrect: bool = False,
    filter_out_nones: bool = False,
):
    print(hue)
    results_dir = f"{exp_dir}/results"
    if "gsm" in exp_dir:
        model = TaskOutput
    else:
        model = StreamingTaskOutput

    outputs = load_per_model_results(Path(results_dir), model, model_names=list(MODEL_NAME_MAP.keys()))

    if filter_bias_on_incorrect:
        outputs = outputs.filter(
            lambda x: x.get_task_spec().get_data_example_obj().biased_ans
            != x.get_task_spec().get_data_example_obj().ground_truth
        )

    df = convert_slist_to_df(outputs, [BasicExtractor(), RawGroundTruthExtractor()])

    df.parsed_response.fillna("None", inplace=True)
    df.parsed_response = df.parsed_response.astype(str)

    # drop where they were none
    if filter_out_nones:
        df = df[df.parsed_response != "None"]

    if "gsm" in exp_dir:
        df["accuracy"] = (df["parsed_response"] == df["raw_ground_truth"]) * 1.0
    else:
        df["accuracy"] = (df["parsed_response"] == df["ground_truth"]) * 1.0

    df["is_biased"] = (df["parsed_response"] == df["biased_ans"]) * 1.0
    df["model"] = df["model"].map(lambda x: MODEL_NAME_MAP[x])  # type: ignore

    # count number of Nones by each type
    print("\nNone counts")
    print(df.groupby(["model", "task_name"]).apply(lambda x: (x.parsed_response == "None").sum()))

    breakpoint()

    name_map = {
        "accuracy": "% Accuracy",
        "model": "Model",
        "is_biased": "% Biased",
        "paraphrasing_formatter_name": "Question Type",
    }

    catplot(
        data=df,
        x="model",
        y="accuracy",
        hue=hue,
        col="label",
        y_scale=100,
        name_map=name_map,
        font_scale=1.5,
    )

    if "gsm" not in exp_dir:
        df2 = df[df["paraphrasing_formatter_name"] != "Original"]
        catplot(
            data=df2,
            x="model",
            y="is_biased",
            hue=hue,
            name_map=name_map,
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
