from abc import abstractmethod
from enum import Enum
from typing import Type, Sequence

from cot_transparency.apis.base import Prompt
from cot_transparency.apis.openai import OpenAIChatPrompt
from slist import Slist, identity

from cot_transparency.data_models.data.biased_question_unbiased_cot import BiasedQuestionUnbiasedCOT
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotUnbiasedFormatter, ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.instructions import VERBALIZE_INSTRUCTION
from cot_transparency.formatters.interventions.few_shots_loading import (
    get_training_cots_gpt_35,
    get_training_non_cots_gpt_35,
    task_output_to_finetune_sample,
    get_training_cots_claude_2,
    get_training_non_cots_claude_2,
)
from cot_transparency.formatters.interventions.big_brain_few_shots_loading import (
    get_training_cots_gpt_35_big_brain,
    get_training_non_cots_gpt_35_big_brain,
    get_training_non_cots_gpt_35_dumb_brain,
    get_training_cots_gpt_35_dumb_brain,
)
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotIgnoreMistakesBiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
)
from cot_transparency.apis.openai.finetune import (
    FinetuneSample,
    FineTuneParams,
    run_finetune_with_wandb,
    FineTuneHyperParams,
)
from scripts.cot_variants import sample_cot_variant
from scripts.load_alpaca_dataset import get_alpaca_training
from scripts.non_cot_variants import non_sample_cot_variant
from scripts.training_formatters import TRAINING_COT_FORMATTERS, TRAINING_NO_COT_FORMATTERS


class Augmentor:
    @staticmethod
    @abstractmethod
    def augment(prompt: Prompt) -> Prompt:
        prompt.model_copy(deep=True)
        return prompt


class RandomCOTPromptAugmentor:
    @staticmethod
    def augment(prompt: Prompt) -> OpenAIChatPrompt:
        new = []
        for message in prompt.messages:
            content: str = message.content
            if VERBALIZE_INSTRUCTION in content:
                content = content.replace(VERBALIZE_INSTRUCTION, sample_cot_variant(content))
            new.append(ChatMessage(role=message.role, content=content))
        return OpenAIChatPrompt(messages=new)


class RandomNonCOTPromptAugmentor:
    @staticmethod
    def augment(prompt: Prompt) -> OpenAIChatPrompt:
        messages = Slist(prompt.messages)
        # ref to the first user message
        first_user_idx: int = messages.find_one_idx_or_raise(lambda x: x.role == MessageRole.user)
        content = messages[first_user_idx].content
        # edit the first user message
        sampled_no_cot_instruction: str = content + "\n" + non_sample_cot_variant(seed=content)
        messages[first_user_idx] = ChatMessage(role=MessageRole.user, content=sampled_no_cot_instruction)

        return OpenAIChatPrompt(messages=messages)


def augment_cots_big_brain(items: Slist[BiasedQuestionUnbiasedCOT]) -> Slist[BiasedQuestionUnbiasedCOT]:
    new = Slist[BiasedQuestionUnbiasedCOT]()
    for item in items:
        new_item = item.model_copy()
        new_item.biased_question = RandomCOTPromptAugmentor.augment(Prompt(messages=item.biased_question)).messages
        # make sure the unbiased context is also augmented
        new_item.unbiased_question = RandomCOTPromptAugmentor.augment(Prompt(messages=item.unbiased_question)).messages
        new.append(new_item)
    return new


def augment_non_cots_big_brain(items: Slist[BiasedQuestionUnbiasedCOT]) -> Slist[BiasedQuestionUnbiasedCOT]:
    new = Slist[BiasedQuestionUnbiasedCOT]()
    for item in items:
        new_item = item.model_copy()
        new_item.biased_question = RandomNonCOTPromptAugmentor.augment(Prompt(messages=item.biased_question)).messages
        # make sure the unbiased context is also augmented
        new_item.unbiased_question = RandomNonCOTPromptAugmentor.augment(
            Prompt(messages=item.unbiased_question)
        ).messages
        new.append(new_item)
    return new


def augment_non_cot(item: FinetuneSample) -> FinetuneSample:
    new_item = item.model_copy()
    non_strict_messages = [m.to_chat_message() for m in item.messages]
    new_item.messages = RandomNonCOTPromptAugmentor.augment(
        OpenAIChatPrompt(messages=non_strict_messages)
    ).get_strict_messages()
    return new_item


def augment_cot(item: FinetuneSample) -> FinetuneSample:
    new_item = item.model_copy()
    non_strict_messages = [m.to_chat_message() for m in item.messages]
    new_item.messages = RandomCOTPromptAugmentor.augment(
        OpenAIChatPrompt(messages=non_strict_messages)
    ).get_strict_messages()
    return new_item


def fine_tune_with_naive_cots(n: int):
    cots: Slist[TaskOutput] = get_training_cots_gpt_35().shuffle(seed="42").take(n)
    print(f"Number of cots: {len(cots)}")
    messages = [FinetuneSample.from_task_output(task) for task in cots]
    params = FineTuneParams(model="gpt-3.5-turbo", hyperparameters=FineTuneHyperParams(n_epochs=1))
    _id = run_finetune_with_wandb(params=params, samples=messages)


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
    print(f"Number of distinct questions: {len(distinct_items)}")
    return (distinct_items.shuffle(seed="42") + non_distinct_items.shuffle(seed="42")).take(limit)


def distinct_at_front_shfufle_big_brain(items: Slist[BiasedQuestionUnbiasedCOT]) -> Slist[BiasedQuestionUnbiasedCOT]:
    shuffled_items = items.shuffle(seed="42")
    already_seen: set[str] = set()
    distinct_items = Slist[BiasedQuestionUnbiasedCOT]()
    non_distinct_items = Slist[BiasedQuestionUnbiasedCOT]()
    for item in shuffled_items:
        if item.original_biased_task.task_spec.task_hash not in already_seen:
            distinct_items.append(item)
            already_seen.add(item.original_biased_task.task_spec.task_hash)
        else:
            non_distinct_items.append(item)
    print(f"Number of distinct items: {len(distinct_items)}")
    return distinct_items.shuffle(seed="42") + non_distinct_items.shuffle(seed="42")


def fine_tune_with_big_brain_cots(
    n: int,
    exclude_formattter: Type[StageOneFormatter] | None,
    n_epochs: int,
    model: str = "gpt-3.5-turbo",
):
    to_exclude_name = exclude_formattter.name() if exclude_formattter is not None else "None"
    data = get_training_non_cots_gpt_35_big_brain() + get_training_cots_gpt_35_big_brain()
    pre_filter = distinct_at_front_shfufle_big_brain(data)
    print(f"Number of cots before filtering: {len(pre_filter)}")
    filtered: Slist[BiasedQuestionUnbiasedCOT] = pre_filter.filter(
        lambda x: x.original_biased_task.task_spec.formatter_name != to_exclude_name
    )
    new = Slist[BiasedQuestionUnbiasedCOT]()
    for item in filtered:
        new_item = item.model_copy()
        new_item.biased_question = RandomCOTPromptAugmentor.augment(Prompt(messages=item.biased_question)).messages
        new.append(new_item)
    print(f"Number of cots after filtering: {len(new)}")
    samples: Slist[FinetuneSample] = (
        new.map(lambda x: x.to_finetune_sample()).shuffle(seed="42").repeat_until_size_or_raise(n)
    )
    print(f"Number of cots: {len(samples)}")
    params = FineTuneParams(model=model, hyperparameters=FineTuneHyperParams(n_epochs=n_epochs))
    _id = run_finetune_with_wandb(params=params, samples=samples)


def fine_tune_with_big_brain_majority_no_cot(
    exclude_formattter: Type[StageOneFormatter] | None,
    n_epochs: int,
    model: str = "gpt-3.5-turbo",
):
    percentage = 0.02
    non_cot_limit = int((1 - percentage) * 72000)
    cot_limit = int(percentage * 72000)
    # 72000 total training
    # 10% aka 7200 are cots, unbiased context, so that the model doesn't forget how to do COTs, but we don't do consistency training on them
    # 90% aka 64800 are non cots, biased context
    to_exclude_name = exclude_formattter.name() if exclude_formattter is not None else "None"
    non_cot = augment_non_cots_big_brain(get_training_non_cots_gpt_35_big_brain()).filter(
        lambda x: x.original_biased_task.task_spec.formatter_name != to_exclude_name
    )
    print(f"Number of non cots: {len(non_cot)}")
    non_cot_limited = non_cot.shuffle("42").repeat_until_size_or_raise(non_cot_limit)
    print(f"Number of non cots after limiting: {len(non_cot_limited)}")
    cot = augment_cots_big_brain(get_training_cots_gpt_35_big_brain()).filter(
        lambda x: x.original_biased_task.task_spec.formatter_name != to_exclude_name
    )
    print(f"Number of cots: {len(cot)}")
    cot_limited = cot.shuffle("42").repeat_until_size_or_raise(cot_limit)
    print(f"Number of cots after limiting: {len(cot_limited)}")
    non_cot_samples = non_cot_limited.map(lambda x: x.to_finetune_sample())
    cot_samples = cot_limited.map(lambda x: x.to_finetune_sample_unbiased_context())
    samples = non_cot_samples + cot_samples + get_alpaca_training(10000)
    params = FineTuneParams(model=model, hyperparameters=FineTuneHyperParams(n_epochs=n_epochs))
    _id = run_finetune_with_wandb(params=params, samples=samples)


def fine_tune_with_big_brain_balanced(
    n_epochs: int,
    exclude_formatters: Sequence[Type[StageOneFormatter]] = [],
    model: str = "gpt-3.5-turbo",
    n_samples: int = 72000,
    instruct_sample_proportion: float = 0.1,
):
    # balanced, all biased context
    percentage = 0.5
    non_cot_limit = int(percentage * n_samples)
    cot_limit = int((1 - percentage) * n_samples)
    excluded_formatters_names = {f.name() for f in exclude_formatters}
    non_cot = augment_non_cots_big_brain(get_training_non_cots_gpt_35_big_brain()).filter(
        lambda x: x.original_biased_task.task_spec.formatter_name not in excluded_formatters_names
        if excluded_formatters_names
        else True
    )
    print(f"Number of non cots: {len(non_cot)}")
    non_cot_limited = non_cot.shuffle("42").repeat_until_size_or_raise(non_cot_limit)
    print(f"Number of non cots after limiting: {len(non_cot_limited)}")
    cot = augment_cots_big_brain(get_training_cots_gpt_35_big_brain()).filter(
        lambda x: x.original_biased_task.task_spec.formatter_name not in excluded_formatters_names
        if excluded_formatters_names
        else True
    )
    print(f"Number of cots: {len(cot)}")
    cot_limited = cot.shuffle("42").repeat_until_size_or_raise(cot_limit)
    print(f"Number of cots after limiting: {len(cot_limited)}")
    non_cot_samples = non_cot_limited.map(lambda x: x.to_finetune_sample())
    cot_samples = cot_limited.map(lambda x: x.to_finetune_sample())
    total_task_samples = non_cot_samples + cot_samples
    n_instruct_samples = int(instruct_sample_proportion * len(total_task_samples))
    alpaca_samples = get_alpaca_training(n_instruct_samples)
    samples = (total_task_samples + alpaca_samples).shuffle("42")
    params = FineTuneParams(model=model, hyperparameters=FineTuneHyperParams(n_epochs=n_epochs))
    more_config = {
        "instruct_sample_proportion": instruct_sample_proportion,
        "n_cots": len(cot_samples),
        "n_non_cots": len(non_cot_samples),
        "n_instruct_samples": len(alpaca_samples),
        "excluded_formatters": list(excluded_formatters_names),
        "brain": "big",
    }
    _id = run_finetune_with_wandb(
        params=params,
        samples=samples,
        notes=f"big brain, balanced 50% cot 50% non cot, {n_samples} samples",
        more_config=more_config,
    )


def sample_from_cot_biases(exclude_formatters: Sequence[Type[StageOneFormatter]]) -> Type[StageOneFormatter]:
    cot_biases = Slist(TRAINING_COT_FORMATTERS)
    return (
        cot_biases.filter(lambda x: x not in exclude_formatters if exclude_formatters else True)
        .shuffle()
        .first_or_raise()
    )


def sample_from_non_cot_biases(
    exclude_formatters: Sequence[Type[StageOneFormatter]], seed: str
) -> Type[StageOneFormatter]:
    non_cot_biases = Slist(TRAINING_NO_COT_FORMATTERS)
    return non_cot_biases.filter(lambda x: x not in exclude_formatters).shuffle(seed=seed).first_or_raise()


def replace_unbiased_cot_prompt_with_biased(
    task: TaskOutput, exclude_formatters: Sequence[Type[StageOneFormatter]]
) -> TaskOutput:
    new = task.model_copy(deep=True)
    assert task.task_spec.formatter_name == ZeroShotCOTUnbiasedFormatter.name()
    sampled_formatter = sample_from_cot_biases(exclude_formatters)
    data_example: DataExampleBase = task.task_spec.get_data_example_obj()
    new.task_spec.messages = sampled_formatter.format_example(data_example)
    return new


def transform_into_post_hoc_reasoning(task: TaskOutput) -> TaskOutput:
    new = task.model_copy(deep=True)
    previous_answer = task.inference_output.parsed_response
    assert previous_answer
    new.inference_output.raw_response = (
        f"The best answer is: ({previous_answer})\n" + task.inference_output.raw_response
    )
    return new


def replace_unbiased_non_cot_prompt_with_biased(
    task: TaskOutput, exclude_formatters: Sequence[Type[StageOneFormatter]], idx: int
) -> TaskOutput:
    new = task.model_copy(deep=True)
    assert task.task_spec.formatter_name == ZeroShotUnbiasedFormatter.name()
    sampled_formatter = sample_from_non_cot_biases(exclude_formatters, seed=str(idx))
    data_example: DataExampleBase = task.task_spec.get_data_example_obj()
    new.task_spec.messages = sampled_formatter.format_example(data_example)
    return new


def clean_unbiased_non_cot_raw_response(task: TaskOutput) -> TaskOutput:
    # Because the model sometimes adds more statements after the answer, and we want to remove it
    assert task.task_spec.formatter_name == ZeroShotUnbiasedFormatter.name()
    new = task.model_copy(deep=True)
    new.inference_output.raw_response = task.task_spec.ground_truth + ")"
    return new


class DataFromOptions(str, Enum):
    gpt_35_turbo = "gpt-3.5-turbo"
    claude_2 = "claude-2"


def fine_tune_with_bias_augmentation_balanced(
    n_epochs: int,
    data_from_options: DataFromOptions = DataFromOptions.gpt_35_turbo,
    exclude_formatters: Sequence[Type[StageOneFormatter]] = [],
    model: str = "gpt-3.5-turbo",
    n_samples: int = 72000,
    instruct_sample_proportion: float = 0.1,
    post_hoc: bool = False,
    cot_percentage=0.5,
) -> str:
    """
    Rather than ensuring that the model changes its answers (big brain),
    we simply use correct COTs as the training data, and add in the biased context
    """
    cot_limit = int(cot_percentage * n_samples)
    non_cot_percentage = 1 - cot_percentage
    non_cot_limit = int((1 - non_cot_percentage) * n_samples)
    excluded_formatters_names = {f.name() for f in exclude_formatters}
    match data_from_options:
        case DataFromOptions.gpt_35_turbo:
            non_cot_data = get_training_non_cots_gpt_35()
            cot_data = get_training_cots_gpt_35()
        case DataFromOptions.claude_2:
            non_cot_data = get_training_non_cots_claude_2()
            cot_data = get_training_cots_claude_2()

    non_cot = non_cot_data.filter(
        lambda x: x.task_spec.formatter_name not in excluded_formatters_names if excluded_formatters_names else True
    )
    print(f"Number of non cots: {len(non_cot)}")
    non_cot_limited = (
        non_cot.shuffle("42")
        .repeat_until_size_or_raise(non_cot_limit)
        .map_enumerate(
            lambda task, idx: replace_unbiased_non_cot_prompt_with_biased(
                task=task, exclude_formatters=exclude_formatters, idx=idx
            )
        )
        .map(clean_unbiased_non_cot_raw_response)
    )
    print(f"Number of non cots after limiting: {len(non_cot_limited)}")
    cot = cot_data.filter(
        lambda x: x.task_spec.formatter_name not in excluded_formatters_names if excluded_formatters_names else True
    )

    print(f"Number of cots: {len(cot)}")
    cot_limited = (
        cot.shuffle("42")
        .repeat_until_size_or_raise(cot_limit)
        .map(lambda x: replace_unbiased_cot_prompt_with_biased(x, exclude_formatters))
        .map(transform_into_post_hoc_reasoning if post_hoc else identity)
    )
    print(f"Number of cots after limiting: {len(cot_limited)}")
    non_cot_samples = non_cot_limited.map(task_output_to_finetune_sample).map(augment_non_cot)
    cot_samples = cot_limited.map(task_output_to_finetune_sample).map(augment_cot)
    total_task_samples = non_cot_samples + cot_samples
    n_instruct_samples = int(instruct_sample_proportion * len(total_task_samples))
    alpaca_samples = get_alpaca_training(n_instruct_samples)
    samples = (total_task_samples + alpaca_samples).shuffle("42")
    params = FineTuneParams(model=model, hyperparameters=FineTuneHyperParams(n_epochs=n_epochs))
    more_config = {
        "instruct_sample_proportion": instruct_sample_proportion,
        "n_cots": len(cot_samples),
        "n_non_cots": len(non_cot_samples),
        "n_instruct_samples": len(alpaca_samples),
        "excluded_formatters": list(excluded_formatters_names),
        "data_from": data_from_options.value,
        "post_hoc": post_hoc,
        "cot_percentage": cot_percentage,
    }
    cot_percentage_percentage = int(cot_percentage * 100)
    non_cot_percentage_percentage = int(non_cot_percentage * 100)
    notes = f"augment unbiased->biased balanced {cot_percentage_percentage}% cot {non_cot_percentage_percentage}% non cot, {n_samples} samples, {data_from_options.value} cots"
    if post_hoc:
        notes = "post hoc " + notes
    _id = run_finetune_with_wandb(
        params=params,
        samples=samples,
        notes=notes,
        more_config=more_config,
    )
    return _id


def fine_tune_with_dumb_brain_balanced(
    n_epochs: int,
    exclude_formatters: Sequence[Type[StageOneFormatter]] = [],
    model: str = "gpt-3.5-turbo",
    n_samples: int = 72000,
    instruct_sample_proportion: float = 0.1,
) -> str:
    # balanced, all biased context
    percentage = 0.5
    non_cot_limit = int(percentage * n_samples)
    cot_limit = int((1 - percentage) * n_samples)
    excluded_formatters_names = {f.name() for f in exclude_formatters}
    non_cot = augment_non_cots_big_brain(get_training_non_cots_gpt_35_dumb_brain()).filter(
        lambda x: x.original_biased_task.task_spec.formatter_name not in excluded_formatters_names
        if excluded_formatters_names
        else True
    )
    print(f"Number of non cots: {len(non_cot)}")
    non_cot_limited = non_cot.shuffle("42").repeat_until_size_or_raise(non_cot_limit)
    print(f"Number of non cots after limiting: {len(non_cot_limited)}")
    cot = augment_cots_big_brain(get_training_cots_gpt_35_dumb_brain()).filter(
        lambda x: x.original_biased_task.task_spec.formatter_name not in excluded_formatters_names
        if excluded_formatters_names
        else True
    )
    print(f"Number of cots: {len(cot)}")
    cot_limited = cot.shuffle("42").repeat_until_size_or_raise(cot_limit)
    print(f"Number of cots after limiting: {len(cot_limited)}")
    non_cot_samples = non_cot_limited.map(lambda x: x.to_finetune_sample())
    cot_samples = cot_limited.map(lambda x: x.to_finetune_sample())
    total_task_samples = non_cot_samples + cot_samples
    n_instruct_samples = int(instruct_sample_proportion * len(total_task_samples))
    alpaca_samples = get_alpaca_training(n_instruct_samples)
    samples = (total_task_samples + alpaca_samples).shuffle("42")
    params = FineTuneParams(model=model, hyperparameters=FineTuneHyperParams(n_epochs=n_epochs))
    more_config = {
        "instruct_sample_proportion": instruct_sample_proportion,
        "n_cots": len(cot_samples),
        "n_non_cots": len(non_cot_samples),
        "n_instruct_samples": len(alpaca_samples),
        "excluded_formatters": list(excluded_formatters_names),
        "brain": "dumb",
    }
    _id = run_finetune_with_wandb(
        params=params,
        samples=samples,
        notes=f"dumb brain, balanced 50% cot 50% non cot, {n_samples} samples",
        more_config=more_config,
    )
    return _id


def fine_tune_with_dumb_brain_balanced_biased_context(
    n_epochs: int,
    exclude_formatters: Sequence[Type[StageOneFormatter]] = [],
    model: str = "gpt-3.5-turbo",
    n_samples: int = 72000,
    instruct_sample_proportion: float = 0.1,
) -> str:
    # balanced, all biased context
    percentage = 0.5
    non_cot_limit = int(percentage * n_samples)
    cot_limit = int((1 - percentage) * n_samples)
    excluded_formatters_names = {f.name() for f in exclude_formatters}
    non_cot = augment_non_cots_big_brain(get_training_non_cots_gpt_35_dumb_brain()).filter(
        lambda x: x.original_biased_task.task_spec.formatter_name not in excluded_formatters_names
        if excluded_formatters_names
        else True
    )
    print(f"Number of non cots: {len(non_cot)}")
    non_cot_limited = non_cot.shuffle("42").repeat_until_size_or_raise(non_cot_limit)
    print(f"Number of non cots after limiting: {len(non_cot_limited)}")
    cot = augment_cots_big_brain(get_training_cots_gpt_35_dumb_brain()).filter(
        lambda x: x.original_biased_task.task_spec.formatter_name not in excluded_formatters_names
        if excluded_formatters_names
        else True
    )
    print(f"Number of cots: {len(cot)}")
    cot_limited = cot.shuffle("42").repeat_until_size_or_raise(cot_limit)
    print(f"Number of cots after limiting: {len(cot_limited)}")
    non_cot_samples = non_cot_limited.map(lambda x: x.to_finetune_sample_using_biased_completion())
    cot_samples = cot_limited.map(lambda x: x.to_finetune_sample_using_biased_completion())
    total_task_samples = non_cot_samples + cot_samples
    n_instruct_samples = int(instruct_sample_proportion * len(total_task_samples))
    alpaca_samples = get_alpaca_training(n_instruct_samples)
    samples = (total_task_samples + alpaca_samples).shuffle("42")
    params = FineTuneParams(model=model, hyperparameters=FineTuneHyperParams(n_epochs=n_epochs))
    more_config = {
        "instruct_sample_proportion": instruct_sample_proportion,
        "n_cots": len(cot_samples),
        "n_non_cots": len(non_cot_samples),
        "n_instruct_samples": len(alpaca_samples),
        "excluded_formatters": list(excluded_formatters_names),
        "brain": "dumb",
        "completion_from": "biased",
    }
    _id = run_finetune_with_wandb(
        params=params,
        samples=samples,
        notes=f"biased completion dumb brain, balanced 50% cot 50% non cot, {n_samples} samples",
        more_config=more_config,
    )
    return _id


def fine_tune_with_big_brain_majority_cot(
    exclude_formattter: Type[StageOneFormatter] | None,
    n_epochs: int,
    model: str = "gpt-3.5-turbo",
):
    percentage = 0.02
    non_cot_limit = int(percentage * 72000)
    cot_limit = int((1 - percentage) * 72000)
    # 72000 total training
    # 10% aka 7200 are non cots, unbiased context, so that the model doesn't forget how to do non COTs, but we don't do consistency training on them
    # 90% aka 64800 are cots, biased context
    to_exclude_name = exclude_formattter.name() if exclude_formattter is not None else "None"
    non_cot = augment_non_cots_big_brain(get_training_non_cots_gpt_35_big_brain()).filter(
        lambda x: x.original_biased_task.task_spec.formatter_name != to_exclude_name
    )
    print(f"Number of non cots: {len(non_cot)}")
    non_cot_limited = non_cot.shuffle("42").repeat_until_size_or_raise(non_cot_limit)
    print(f"Number of non cots after limiting: {len(non_cot_limited)}")
    cot = augment_cots_big_brain(get_training_cots_gpt_35_big_brain()).filter(
        lambda x: x.original_biased_task.task_spec.formatter_name != to_exclude_name
    )
    print(f"Number of cots: {len(cot)}")
    cot_limited = cot.shuffle("42").repeat_until_size_or_raise(cot_limit)
    print(f"Number of cots after limiting: {len(cot_limited)}")
    non_cot_samples = non_cot_limited.map(lambda x: x.to_finetune_sample_unbiased_context())
    cot_samples = cot_limited.map(lambda x: x.to_finetune_sample())
    alpaca_samples = get_alpaca_training(10000)
    samples = (non_cot_samples + cot_samples + alpaca_samples).shuffle("42")
    params = FineTuneParams(model=model, hyperparameters=FineTuneHyperParams(n_epochs=n_epochs))
    _id = run_finetune_with_wandb(params=params, samples=samples)


def fine_tune_with_unbiased_majority_cot(
    exclude_formattter: Type[StageOneFormatter] | None,
    n_epochs: int,
    model: str = "gpt-3.5-turbo",
):
    percentage = 0.02
    non_cot_limit = int(percentage * 72000)
    cot_limit = int((1 - percentage) * 72000)
    # 72000 total training
    # 10% aka 7200 are non cots, unbiased context, so that the model doesn't forget how to do non COTs, but we don't do consistency training on them
    # 90% aka 64800 are cots, biased context
    to_exclude_name = exclude_formattter.name() if exclude_formattter is not None else "None"
    non_cot = augment_non_cots_big_brain(get_training_non_cots_gpt_35_big_brain()).filter(
        lambda x: x.original_biased_task.task_spec.formatter_name != to_exclude_name
    )
    print(f"Number of non cots: {len(non_cot)}")
    non_cot_limited = non_cot.shuffle("42").repeat_until_size_or_raise(non_cot_limit)
    print(f"Number of non cots after limiting: {len(non_cot_limited)}")
    cot = augment_cots_big_brain(get_training_cots_gpt_35_big_brain()).filter(
        lambda x: x.original_biased_task.task_spec.formatter_name != to_exclude_name
    )
    print(f"Number of cots: {len(cot)}")
    cot_limited = cot.shuffle("42").repeat_until_size_or_raise(cot_limit)
    print(f"Number of cots after limiting: {len(cot_limited)}")
    non_cot_samples = non_cot_limited.map(lambda x: x.to_finetune_sample_unbiased_context())
    cot_samples = cot_limited.map(lambda x: x.to_finetune_sample_unbiased_context())
    alpaca_samples = get_alpaca_training(10000)
    samples = (non_cot_samples + cot_samples + alpaca_samples).shuffle("42")
    params = FineTuneParams(model=model, hyperparameters=FineTuneHyperParams(n_epochs=n_epochs))
    _id = run_finetune_with_wandb(params=params, samples=samples)


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
    _id = run_finetune_with_wandb(params=params, samples=samples)


if __name__ == "__main__":
    fine_tune_with_dumb_brain_balanced_biased_context(
        n_epochs=1,
        exclude_formatters=[WrongFewShotIgnoreMistakesBiasedFormatter, WrongFewShotIgnoreMistakesBiasedNoCOTFormatter],
        n_samples=12000,
    )
    fine_tune_with_dumb_brain_balanced(
        n_epochs=1,
        exclude_formatters=[WrongFewShotIgnoreMistakesBiasedFormatter, WrongFewShotIgnoreMistakesBiasedNoCOTFormatter],
        n_samples=100,
    )
    fine_tune_with_big_brain_balanced(
        model="gpt-3.5-turbo",
        n_epochs=1,
        exclude_formatters=[WrongFewShotIgnoreMistakesBiasedFormatter, WrongFewShotIgnoreMistakesBiasedNoCOTFormatter],
        n_samples=100,
    )
    fine_tune_with_bias_augmentation_balanced(
        model="gpt-3.5-turbo",
        n_epochs=1,
        exclude_formatters=[WrongFewShotIgnoreMistakesBiasedFormatter, WrongFewShotIgnoreMistakesBiasedNoCOTFormatter],
        n_samples=100,
        data_from_options=DataFromOptions.gpt_35_turbo,
    )
    # fine_tune_with_big_brain_majority_cot(n_epochs=1, exclude_formattter=WrongFewShotIgnoreMistakesBiasedFormatter)
    # fine_tune_with_unbiased_majority_cot(n_epochs=1, exclude_formattter=WrongFewShotIgnoreMistakesBiasedFormatter)
    # fine_tune_with_big_brain_majority_no_cot(
    #     n_epochs=1, exclude_formattter=WrongFewShotIgnoreMistakesBiasedNoCOTFormatter
    # )
