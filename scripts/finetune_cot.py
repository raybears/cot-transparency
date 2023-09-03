from typing import Type, Sequence

from pydantic import ValidationError
from slist import Slist

from cot_transparency.data_models.data import ArcExample
from cot_transparency.data_models.data.bbh import MilesBBHRawData
from cot_transparency.data_models.data.biased_question_unbiased_cot import BiasedQuestionUnbiasedCOT
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter
from cot_transparency.formatters.interventions.few_shots_loading import (
    get_training_cots_gpt_35,
)
from cot_transparency.formatters.interventions.big_brain_few_shots_loading import get_training_cots_gpt_35_big_brain
from cot_transparency.formatters.interventions.formatting import get_formatter_for_few_shot_cot
from cot_transparency.formatters.more_biases.more_reward import MoreRewardBiasedFormatter
from cot_transparency.formatters.more_biases.wrong_few_shot import WrongFewShotIgnoreMistakesBiasedFormatter
from cot_transparency.formatters.verbalize.formatters import (
    StanfordBiasedFormatter,
    CheckmarkBiasedFormatter,
    CrossBiasedFormatter,
)
from cot_transparency.model_apis import Prompt, ModelType
from cot_transparency.openai_utils.finetune import (
    FinetuneSample,
    FineTuneParams,
    run_finetune,
    FineTuneHyperParams,
    join_assistant_preferred_to_completion,
)


def task_output_to_biased_question_with_correct_answer(
    task: TaskOutput,
    exclude_formattter: Type[StageOneFormatter] | None,
    idx: int,
    use_formatters: Sequence[Type[StageOneFormatter]] = Slist(),
) -> FinetuneSample:
    try:
        read = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    except ValidationError:
        read = task.task_spec.read_data_example_or_raise(ArcExample)
    formatter_to_use = get_formatter_for_few_shot_cot(
        exclude_formattter=exclude_formattter, seed=read.hash() + str(idx), use_formatters=use_formatters
    )
    prompt_messages = formatter_to_use.format_example(read)
    joined = join_assistant_preferred_to_completion(
        messages=prompt_messages, completion=task.inference_output.raw_response
    )
    strict = Prompt(messages=joined)
    return FinetuneSample(messages=strict.get_strict_messages(ModelType.chat))


def fine_tune_with_naive_cots(n: int):
    cots: Slist[TaskOutput] = get_training_cots_gpt_35().shuffle(seed="42").take(n)
    print(f"Number of cots: {len(cots)}")
    messages = [FinetuneSample.from_task_output(task) for task in cots]
    params = FineTuneParams(model="gpt-3.5-turbo", hyperparameters=FineTuneHyperParams(n_epochs=1))
    _id = run_finetune(params=params, samples=messages)


def distinct_at_front_shuffle(items: Slist[TaskOutput], limit: int) -> Slist[TaskOutput]:
    """Shuffles the items, but puts the distinct task hash items at the front"""
    already_seen: set[str] = set()
    distinct_items = Slist[TaskOutput]()
    non_distinct_items = Slist[TaskOutput]()
    for item in items:
        if item.task_spec.task_hash not in already_seen:
            distinct_items.append(item)
            already_seen.add(item.task_spec.task_hash)
        else:
            non_distinct_items.append(item)
    print(f"Number of distinct items: {len(distinct_items)}")
    return (distinct_items.shuffle(seed="42") + non_distinct_items.shuffle(seed="42")).take(limit)


def distinct_at_front_shfufle_big_brain(items: Slist[BiasedQuestionUnbiasedCOT]) -> Slist[BiasedQuestionUnbiasedCOT]:
    already_seen: set[str] = set()
    distinct_items = Slist[BiasedQuestionUnbiasedCOT]()
    non_distinct_items = Slist[BiasedQuestionUnbiasedCOT]()
    for item in items:
        if item.original_biased_task.task_spec.task_hash not in already_seen:
            distinct_items.append(item)
            already_seen.add(item.original_biased_task.task_spec.task_hash)
        else:
            non_distinct_items.append(item)
    print(f"Number of distinct items: {len(distinct_items)}")
    return distinct_items.shuffle(seed="42") + non_distinct_items.shuffle(seed="42")


def fine_tune_with_biased_cots(
    n: int,
    exclude_formattter: Type[StageOneFormatter] | None,
    use_formatters: Sequence[Type[StageOneFormatter]],
    n_epochs: int,
    model: str = "gpt-3.5-turbo",
):
    cots: Slist[TaskOutput] = distinct_at_front_shuffle(
        items=get_training_cots_gpt_35(), limit=n
    ).repeat_until_size_or_raise(n)
    print(f"Number of cots: {len(cots)}")
    messages = [
        task_output_to_biased_question_with_correct_answer(
            task, exclude_formattter=exclude_formattter, use_formatters=use_formatters, idx=idx
        )
        for idx, task in enumerate(cots)
    ]
    params = FineTuneParams(model=model, hyperparameters=FineTuneHyperParams(n_epochs=n_epochs))
    _id = run_finetune(params=params, samples=messages)


def fine_tune_with_big_brain_cots(
    n: int,
    exclude_formattter: Type[StageOneFormatter] | None,
    n_epochs: int,
    model: str = "gpt-3.5-turbo",
):
    to_exclude_name = exclude_formattter.name() if exclude_formattter is not None else "None"
    pre_filter = distinct_at_front_shfufle_big_brain(get_training_cots_gpt_35_big_brain())
    print(f"Number of cots before filtering: {len(pre_filter)}")
    filtered = pre_filter.filter(lambda x: x.original_biased_task.task_spec.formatter_name != to_exclude_name)
    print(f"Number of cots after filtering: {len(filtered)}")
    samples: Slist[FinetuneSample] = filtered.map(lambda x: x.to_finetune_sample()).repeat_until_size_or_raise(n)
    print(f"Number of cots: {len(samples)}")
    params = FineTuneParams(model=model, hyperparameters=FineTuneHyperParams(n_epochs=n_epochs))
    _id = run_finetune(params=params, samples=samples)


def fine_tune_with_big_brain_cots_control_tokens(
    n: int,
    exclude_formattter: Type[StageOneFormatter] | None,
    n_epochs: int,
    model: str = "gpt-3.5-turbo",
):
    to_exclude_name = exclude_formattter.name() if exclude_formattter is not None else "None"
    pre_filter: Slist[BiasedQuestionUnbiasedCOT] = distinct_at_front_shfufle_big_brain(
        get_training_cots_gpt_35_big_brain()
    )
    print(f"Number of cots before filtering: {len(pre_filter)}")
    filtered = pre_filter.filter(lambda x: x.original_biased_task.task_spec.formatter_name != to_exclude_name)
    print(f"Number of cots after filtering: {len(filtered)}")
    samples: Slist[FinetuneSample] = (
        filtered.map(lambda x: x.to_finetune_sample_control_tokens()).flatten_list().repeat_until_size_or_raise(n * 2)
    )
    print(f"Number of cots: {len(samples)}")
    params = FineTuneParams(model=model, hyperparameters=FineTuneHyperParams(n_epochs=n_epochs))
    _id = run_finetune(params=params, samples=samples)


if __name__ == "__main__":
    use_formatters = Slist(
        [
            ZeroShotCOTSycophancyFormatter,
            StanfordBiasedFormatter,
            WrongFewShotIgnoreMistakesBiasedFormatter,
            MoreRewardBiasedFormatter,
            CheckmarkBiasedFormatter,
            CrossBiasedFormatter,
        ]
    )
    fine_tune_with_big_brain_cots_control_tokens(
        6000,
        exclude_formattter=WrongFewShotIgnoreMistakesBiasedFormatter,
        n_epochs=1,
        model="gpt-3.5-turbo",
    )
    # fine_tune_with_biased_cots(
    #     72000,
    #     exclude_formattter=WrongFewShotIgnoreMistakesBiasedFormatter,
    #     use_formatters=use_formatters,
    #     n_epochs=1,
    #     model="gpt-3.5-turbo",
    # )
    # fine_tune_with_biased_cots(6000, exclude_formattter=WrongFewShotIgnoreMistakesBiasedFormatter, use_formatters=use_formatters, n_epochs=1)
