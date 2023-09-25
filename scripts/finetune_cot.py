from abc import abstractmethod
from typing import Type

from slist import Slist

from cot_transparency.data_models.data.biased_question_unbiased_cot import BiasedQuestionUnbiasedCOT
from cot_transparency.data_models.models import TaskOutput, ChatMessage, MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter
from cot_transparency.formatters.instructions import VERBALIZE_INSTRUCTION
from cot_transparency.formatters.interventions.few_shots_loading import (
    get_training_cots_gpt_35,
)
from cot_transparency.formatters.interventions.big_brain_few_shots_loading import (
    get_training_cots_gpt_35_big_brain,
    get_training_non_cots_gpt_35_big_brain, get_training_non_cots_gpt_35_dumb_brain,
    get_training_cots_gpt_35_dumb_brain,
)
from cot_transparency.formatters.more_biases.more_reward import MoreRewardBiasedFormatter
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotIgnoreMistakesBiasedFormatter,
)
from cot_transparency.formatters.verbalize.formatters import (
    StanfordBiasedFormatter,
    CheckmarkBiasedFormatter,
    CrossBiasedFormatter,
)
from cot_transparency.model_apis import Prompt
from cot_transparency.openai_utils.finetune import (
    FinetuneSample,
    FineTuneParams,
    run_finetune,
    FineTuneHyperParams,
)
from scripts.cot_variants import sample_cot_variant
from scripts.load_alpaca_dataset import get_alpaca_training
from scripts.non_cot_variants import non_sample_cot_variant


class Augmentor:
    @staticmethod
    @abstractmethod
    def augment(prompt: Prompt) -> Prompt:
        prompt.model_copy(deep=True)
        return prompt


class RandomCOTPromptAugmentor:
    @staticmethod
    def augment(prompt: Prompt) -> Prompt:
        new = []
        for message in prompt.messages:
            content: str = message.content
            if VERBALIZE_INSTRUCTION in content:
                content = content.replace(VERBALIZE_INSTRUCTION, sample_cot_variant(content))
            new.append(ChatMessage(role=message.role, content=content))
        return Prompt(messages=new)


class RandomNonCOTPromptAugmentor:
    @staticmethod
    def augment(prompt: Prompt) -> Prompt:
        messages = Slist(prompt.messages)
        # ref to the first user message
        first_user_idx: int = messages.find_one_idx_or_raise(lambda x: x.role == MessageRole.user)
        content = messages[first_user_idx].content
        # edit the first user message
        sampled_no_cot_instruction: str = content + "\n" + non_sample_cot_variant(seed=content)
        messages[first_user_idx] = ChatMessage(role=MessageRole.user, content=sampled_no_cot_instruction)

        return Prompt(messages=messages)


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
    _id = run_finetune(params=params, samples=samples)


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
    _id = run_finetune(params=params, samples=samples)


def fine_tune_with_big_brain_balanced(
    exclude_formattter: Type[StageOneFormatter] | None,
    n_epochs: int,
    model: str = "gpt-3.5-turbo",
):
    # balanced, all biased context
    percentage = 0.5
    non_cot_limit = int(percentage * 72000)
    cot_limit = int((1 - percentage) * 72000)
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
    cot_samples = cot_limited.map(lambda x: x.to_finetune_sample())
    alpaca_samples = get_alpaca_training(10000)
    samples = (non_cot_samples + cot_samples + alpaca_samples).shuffle("42")
    params = FineTuneParams(model=model, hyperparameters=FineTuneHyperParams(n_epochs=n_epochs))
    _id = run_finetune(params=params, samples=samples)

def fine_tune_with_dumb_brain_balanced(
    exclude_formattter: Type[StageOneFormatter] | None,
    n_epochs: int,
    model: str = "gpt-3.5-turbo",
):
    # balanced, all biased context
    percentage = 0.5
    non_cot_limit = int(percentage * 72000)
    cot_limit = int((1 - percentage) * 72000)
    to_exclude_name = exclude_formattter.name() if exclude_formattter is not None else "None"
    non_cot = augment_non_cots_big_brain(get_training_non_cots_gpt_35_dumb_brain()).filter(
        lambda x: x.original_biased_task.task_spec.formatter_name != to_exclude_name
    )
    print(f"Number of non cots: {len(non_cot)}")
    non_cot_limited = non_cot.shuffle("42").repeat_until_size_or_raise(non_cot_limit)
    print(f"Number of non cots after limiting: {len(non_cot_limited)}")
    cot = augment_cots_big_brain(get_training_cots_gpt_35_dumb_brain()).filter(
        lambda x: x.original_biased_task.task_spec.formatter_name != to_exclude_name
    )
    print(f"Number of cots: {len(cot)}")
    cot_limited = cot.shuffle("42").repeat_until_size_or_raise(cot_limit)
    print(f"Number of cots after limiting: {len(cot_limited)}")
    non_cot_samples = non_cot_limited.map(lambda x: x.to_finetune_sample())
    cot_samples = cot_limited.map(lambda x: x.to_finetune_sample())
    alpaca_samples = get_alpaca_training(10000)
    samples = (non_cot_samples + cot_samples + alpaca_samples).shuffle("42")
    params = FineTuneParams(model=model, hyperparameters=FineTuneHyperParams(n_epochs=n_epochs))
    _id = run_finetune(params=params, samples=samples)

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
    _id = run_finetune(params=params, samples=samples)


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
    fine_tune_with_dumb_brain_balanced(n_epochs=1, exclude_formattter=WrongFewShotIgnoreMistakesBiasedFormatter)
    # fine_tune_with_big_brain_balanced(n_epochs=1, exclude_formattter=WrongFewShotIgnoreMistakesBiasedFormatter)
    # fine_tune_with_big_brain_majority_cot(n_epochs=1, exclude_formattter=WrongFewShotIgnoreMistakesBiasedFormatter)
    # fine_tune_with_unbiased_majority_cot(n_epochs=1, exclude_formattter=WrongFewShotIgnoreMistakesBiasedFormatter)
    # fine_tune_with_big_brain_majority_no_cot(
    #     n_epochs=1, exclude_formattter=WrongFewShotIgnoreMistakesBiasedNoCOTFormatter
    # )
