import dataclasses
import json
import math
import random
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Sequence
from enum import Enum

from slist import Slist, identity

from cot_transparency.apis.base import Prompt
from cot_transparency.apis.openai import OpenAIChatPrompt
from cot_transparency.apis.openai.finetune import (
    FineTuneHyperParams,
    FineTuneParams,
    FinetuneSample,
    run_finetune_with_wandb,
)
from cot_transparency.data_models.data.biased_question_unbiased_cot import (
    BiasedQuestionUnbiasedCOT,
)
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.unbiased import (
    ZeroShotCOTUnbiasedFormatter,
    ZeroShotUnbiasedFormatter,
)
from cot_transparency.formatters.instructions import VERBALIZE_INSTRUCTION
from cot_transparency.formatters.interventions.big_brain_few_shots_loading import (
    get_training_cots_gpt_35_big_brain,
    get_training_cots_gpt_35_dumb_brain,
    get_training_non_cots_gpt_35_big_brain,
    get_training_non_cots_gpt_35_dumb_brain,
)
from cot_transparency.formatters.interventions.few_shots_loading import (
    ModelOutputVerified,
    get_training_cots_claude_2,
    get_training_cots_gpt_35,
    get_training_non_cots_claude_2,
    get_training_non_cots_gpt_35,
    task_output_to_finetune_sample,
)
from cot_transparency.formatters.interventions.intervention import Intervention
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotIgnoreMistakesBiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
)
from cot_transparency.formatters.prompt_sensitivity.interventions import (
    AddBestAnswerIsNonCot,
    AddVerbalizeAndStepByStepAssistantPref,
)
from cot_transparency.formatters.prompt_sensitivity.v2_prompt_sen import (
    TRAINING_COT_PROMPT_VARIANTS_8,
    TRAINING_COT_PROMPT_VARIANTS_ALL,
    TRAINING_NO_COT_PROMPT_VARIANTS_7,
    TRAINING_NO_COT_PROMPT_VARIANTS_ALL,
)
from scripts.cot_variants import sample_cot_variant
from scripts.load_alpaca_dataset import get_alpaca_training
from scripts.non_cot_variants import non_sample_cot_variant
from scripts.training_formatters import (
    TRAINING_COT_FORMATTERS,
    TRAINING_COT_FORMATTERS_FEW_SHOT,
    TRAINING_COT_FORMATTERS_ZERO_SHOT,
    TRAINING_NO_COT_FORMATTERS,
    TRAINING_NO_COT_FORMATTERS_FEW_SHOT,
    TRAINING_NO_COT_FORMATTERS_ZERO_SHOT,
)


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


def augment_cots_big_brain(
    items: Slist[BiasedQuestionUnbiasedCOT],
) -> Slist[BiasedQuestionUnbiasedCOT]:
    new = Slist[BiasedQuestionUnbiasedCOT]()
    for item in items:
        new_item = item.model_copy()
        new_item.biased_question = RandomCOTPromptAugmentor.augment(Prompt(messages=item.biased_question)).messages
        # make sure the unbiased context is also augmented
        new_item.unbiased_question = RandomCOTPromptAugmentor.augment(Prompt(messages=item.unbiased_question)).messages
        new.append(new_item)
    return new


def augment_non_cots_big_brain(
    items: Slist[BiasedQuestionUnbiasedCOT],
) -> Slist[BiasedQuestionUnbiasedCOT]:
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


def augment_non_cot_task(item: TaskOutput) -> TaskOutput:
    new_messages = RandomNonCOTPromptAugmentor.augment(OpenAIChatPrompt(messages=item.task_spec.messages)).messages

    return item.copy_update(task_spec=item.task_spec.copy_update(messages=new_messages))


def augment_cot_task(item: TaskOutput) -> TaskOutput:
    new_messages = RandomCOTPromptAugmentor.augment(OpenAIChatPrompt(messages=item.task_spec.messages)).messages
    return item.copy_update(task_spec=item.task_spec.copy_update(messages=new_messages))


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


def distinct_at_front_shfufle_big_brain(
    items: Slist[BiasedQuestionUnbiasedCOT],
) -> Slist[BiasedQuestionUnbiasedCOT]:
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


def fine_tune_with_big_brain(
    n_epochs: int,
    exclude_formatters: Sequence[type[StageOneFormatter]] = [],
    model: str = "gpt-3.5-turbo",
    n_samples: int = 72000,
    instruct_sample_proportion: float = 0.1,
    cot_proportion: float = 0.5,
) -> str:
    non_cot_limit = int((1 - cot_proportion) * n_samples)
    cot_limit = int(cot_proportion * n_samples)
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
        notes=f"big brained, cot_proportion={cot_proportion}",
        more_config=more_config,
    )
    return _id


def sample_from_cot_biases(
    exclude_formatters: Sequence[type[StageOneFormatter]],
) -> type[StageOneFormatter]:
    cot_biases = Slist(TRAINING_COT_FORMATTERS)
    return (
        cot_biases.filter(lambda x: x not in exclude_formatters if exclude_formatters else True)
        .shuffle()
        .first_or_raise()
    )


def sample_from_non_cot_biases(
    exclude_formatters: Sequence[type[StageOneFormatter]], seed: str
) -> type[StageOneFormatter]:
    non_cot_biases = Slist(TRAINING_NO_COT_FORMATTERS)
    return non_cot_biases.filter(lambda x: x not in exclude_formatters).shuffle(seed=seed).first_or_raise()


def replace_unbiased_cot_prompt_with_biased(
    task: TaskOutput, exclude_formatters: Sequence[type[StageOneFormatter]]
) -> TaskOutput:
    new = task.model_copy(deep=True)
    assert task.task_spec.formatter_name == ZeroShotCOTUnbiasedFormatter.name()
    sampled_formatter = sample_from_cot_biases(exclude_formatters)
    data_example: DataExampleBase = task.task_spec.get_data_example_obj()
    new.task_spec.messages = sampled_formatter.format_example(data_example)
    return new


def replace_unbiased_cot_prompt_with_formatters(
    task: TaskOutput,
    use_formatters: Iterable[type[StageOneFormatter]],
    intervention: type[Intervention] | None = None,
) -> Slist[TaskOutput]:
    output = Slist[TaskOutput]()
    for formatter in use_formatters:
        new = task.model_copy(deep=True)

        assert (
            task.task_spec.formatter_name == ZeroShotCOTUnbiasedFormatter.name()
        ), f"Got {task.task_spec.formatter_name}"
        data_example: DataExampleBase = task.task_spec.get_data_example_obj()
        if intervention is not None:
            new.task_spec.messages = intervention.intervene(question=data_example, formatter=formatter)
        else:
            new.task_spec.messages = formatter.format_example(data_example)
        output.append(new)
    return output


def transform_into_post_hoc_reasoning(task: TaskOutput) -> TaskOutput:
    new = task.model_copy(deep=True)
    previous_answer = task.inference_output.parsed_response
    assert previous_answer
    new.inference_output.raw_response = (
        f"The best answer is: ({previous_answer})\n" + task.inference_output.raw_response
    )
    return new


@dataclasses.dataclass(kw_only=True, frozen=True)
class FormatterWithPossibleIntervention:
    formatter: type[StageOneFormatter]
    intervention: type[Intervention] | None = None

    def name(self):
        return self.formatter.name() + (f"_{self.intervention.name()}" if self.intervention else "")

    def __lt__(self, other: "FormatterWithPossibleIntervention"):
        if not isinstance(other, FormatterWithPossibleIntervention):
            return NotImplemented
        # Compare the names of the formatters and interventions, not the classes themselves
        return self.name() < other.name()


def replace_unbiased_prompt_with_formatters(
    task: TaskOutput,
    use_formatters: Iterable[FormatterWithPossibleIntervention],
) -> Slist[TaskOutput]:
    output = Slist[TaskOutput]()
    for fwpi in use_formatters:
        new = task
        data_example: DataExampleBase = task.task_spec.get_data_example_obj()
        intervention = fwpi.intervention
        new_messages = (
            fwpi.formatter.format_example(data_example)
            if intervention is None
            else intervention.intervene(
                question=data_example,
                formatter=fwpi.formatter,
                model=task.task_spec.inference_config.model,
            )
        )
        new = new.copy_update(task_spec=new.task_spec.copy_update(messages=new_messages))
        output.append(new)
    return output


class DataFromOptions(str, Enum):
    gpt_35_turbo = "gpt-3.5-turbo"
    claude_2 = "claude-2"


class FormatterOptions(str, Enum):
    # What types of formatters to use
    # see match_formatter_options for details
    control_only_unbiased = "control_only_unbiased"
    all_biased = "all_biased"
    zero_shot = "zero_shot"
    few_shot = "few_shot"
    prompt_variants_set1 = "prompt_variants_set1"
    prompt_variants_all = "prompt_variants_all"
    super_dataset = "super_dataset"


@dataclasses.dataclass(kw_only=True)
class FormatterOptionsResult:
    biased_formatters: Sequence[FormatterWithPossibleIntervention]
    unbiased_formatters: Sequence[FormatterWithPossibleIntervention]


def match_formatter_options(
    formatter_options: FormatterOptions,
) -> FormatterOptionsResult:
    non_cot_formatters: Sequence[FormatterWithPossibleIntervention]
    cot_formatters: Sequence[FormatterWithPossibleIntervention]

    match formatter_options:
        case FormatterOptions.all_biased:
            non_cot_formatters = Slist(TRAINING_NO_COT_FORMATTERS).map(
                lambda x: FormatterWithPossibleIntervention(formatter=x)
            )
            cot_formatters = Slist(TRAINING_COT_FORMATTERS).map(
                lambda x: FormatterWithPossibleIntervention(formatter=x)
            )
        case FormatterOptions.zero_shot:
            non_cot_formatters = Slist(TRAINING_NO_COT_FORMATTERS_ZERO_SHOT).map(
                lambda x: FormatterWithPossibleIntervention(formatter=x)
            )
            cot_formatters = Slist(TRAINING_COT_FORMATTERS_ZERO_SHOT).map(
                lambda x: FormatterWithPossibleIntervention(formatter=x)
            )
        case FormatterOptions.few_shot:
            non_cot_formatters = Slist(TRAINING_NO_COT_FORMATTERS_FEW_SHOT).map(
                lambda x: FormatterWithPossibleIntervention(formatter=x)
            )
            cot_formatters = Slist(TRAINING_COT_FORMATTERS_FEW_SHOT).map(
                lambda x: FormatterWithPossibleIntervention(formatter=x)
            )
        case FormatterOptions.control_only_unbiased:
            non_cot_formatters = [FormatterWithPossibleIntervention(formatter=ZeroShotUnbiasedFormatter)]
            cot_formatters = [FormatterWithPossibleIntervention(formatter=ZeroShotCOTUnbiasedFormatter)]
        case FormatterOptions.prompt_variants_set1:
            non_cot_formatters = TRAINING_NO_COT_PROMPT_VARIANTS_7.map(
                lambda x: FormatterWithPossibleIntervention(formatter=x, intervention=AddBestAnswerIsNonCot)
            )
            cot_formatters = TRAINING_COT_PROMPT_VARIANTS_8.map(
                lambda x: FormatterWithPossibleIntervention(
                    formatter=x,
                    intervention=AddVerbalizeAndStepByStepAssistantPref,
                )
            )
        case FormatterOptions.prompt_variants_all:
            cot_formatters = TRAINING_COT_PROMPT_VARIANTS_ALL.map(
                lambda x: FormatterWithPossibleIntervention(
                    formatter=x,
                    intervention=AddVerbalizeAndStepByStepAssistantPref,
                )
            )
            non_cot_formatters = TRAINING_NO_COT_PROMPT_VARIANTS_ALL.map(
                lambda x: FormatterWithPossibleIntervention(formatter=x, intervention=AddBestAnswerIsNonCot)
            )
        case FormatterOptions.super_dataset:
            # this is the same as prompt_variants_all + zero shot + few shot
            cot_formatters = (
                TRAINING_COT_PROMPT_VARIANTS_ALL.map(
                    lambda x: FormatterWithPossibleIntervention(
                        formatter=x,
                        intervention=AddVerbalizeAndStepByStepAssistantPref,
                    )
                )
                + Slist(TRAINING_COT_FORMATTERS_ZERO_SHOT).map(lambda x: FormatterWithPossibleIntervention(formatter=x))
                + Slist(TRAINING_COT_FORMATTERS_FEW_SHOT).map(lambda x: FormatterWithPossibleIntervention(formatter=x))
            )
            non_cot_formatters = (
                TRAINING_NO_COT_PROMPT_VARIANTS_ALL.map(
                    lambda x: FormatterWithPossibleIntervention(formatter=x, intervention=AddBestAnswerIsNonCot)
                )
                + Slist(TRAINING_NO_COT_FORMATTERS_ZERO_SHOT).map(
                    lambda x: FormatterWithPossibleIntervention(formatter=x)
                )
                + Slist(TRAINING_NO_COT_FORMATTERS_FEW_SHOT).map(
                    lambda x: FormatterWithPossibleIntervention(formatter=x)
                )
            )

    return FormatterOptionsResult(
        biased_formatters=sorted(list(set(cot_formatters))),
        unbiased_formatters=sorted(list(set(non_cot_formatters))),
    )


def fine_tune_with_bias_augmentation_no_repeat(
    n_epochs: int,
    data_from_options: DataFromOptions = DataFromOptions.gpt_35_turbo,
    model_output_verified: ModelOutputVerified = ModelOutputVerified.correct,
    exclude_formatters: Sequence[type[StageOneFormatter]] = [],
    # if FormatterOptions.control_only_unbiased, then we only use unbiased contexts for training
    formatter_options: FormatterOptions = FormatterOptions.all_biased,
    project_name: str = "consistency-training",
    model: str = "gpt-3.5-turbo",
    n_samples: int = 72000,
    instruct_sample_proportion: float = 0.1,
    post_hoc: bool = False,
    cot_percentage=0.5,
    # cli waits for user input to validate the training
    ask_to_validate_training: bool = True,
) -> str:
    """
    We use unbiased correct COTs, then replace the unbiased COT prompt with a biased COT formatter prompt
    """
    cot_limit = int(cot_percentage * n_samples)
    non_cot_percentage = 1 - cot_percentage
    non_cot_limit = int(non_cot_percentage * n_samples)
    excluded_formatters_names = {f.name() for f in exclude_formatters}

    match data_from_options:
        case DataFromOptions.gpt_35_turbo:
            non_cot_data = get_training_non_cots_gpt_35(model_output_verified)
            cot_data = get_training_cots_gpt_35(model_output_verified)
        case DataFromOptions.claude_2:
            non_cot_data = get_training_non_cots_claude_2(model_output_verified)
            cot_data = get_training_cots_claude_2(model_output_verified)

    non_cot_data_shuffled = non_cot_data.shuffle(seed="42")
    cot_data_shuffled = cot_data.shuffle(seed="42")
    formatter_options_result = match_formatter_options(formatter_options)

    non_cot_formatters = formatter_options_result.unbiased_formatters
    cot_formatters = formatter_options_result.biased_formatters

    eligible_non_cot_formatters = Slist(non_cot_formatters).filter(lambda x: x.formatter not in exclude_formatters)
    assert len(eligible_non_cot_formatters) > 0, "We do not have any eligible non cot formatters"
    eligible_cot_formatters = Slist(cot_formatters).filter(lambda x: x.formatter not in exclude_formatters)
    assert len(eligible_cot_formatters) > 0, "We do not have any eligible cot formatters"

    print(f"Number of non cots: {len(non_cot_data)}")
    non_cot_limited = non_cot_data_shuffled.map(
        lambda task: replace_unbiased_prompt_with_formatters(
            task=task,
            use_formatters=eligible_non_cot_formatters,
        )
        .shuffle()
        .first_or_raise()
    ).take(non_cot_limit)
    assert (
        len(non_cot_limited) == non_cot_limit
    ), f"We do not have enough non cots, only {len(non_cot_limited)}, required {non_cot_limit}"
    print(f"Number of non cots after limiting: {len(non_cot_limited)}")

    print(f"Number of cots: {len(cot_data)}")
    cot_limited = (
        cot_data_shuffled.map(
            lambda task: replace_unbiased_prompt_with_formatters(task=task, use_formatters=eligible_cot_formatters)
            .shuffle()
            .first_or_raise()
        )
        .map(transform_into_post_hoc_reasoning if post_hoc else identity)
        .take(cot_limit)
    )
    assert len(cot_limited) == cot_limit, f"We do not have enough cots, only {len(cot_limited)}"
    print(f"Number of cots after limiting: {len(cot_limited)}")
    non_cot_samples = non_cot_limited.map(augment_non_cot_task).map(task_output_to_finetune_sample)
    cot_samples = cot_limited.map(augment_cot_task).map(task_output_to_finetune_sample)
    total_task_samples = non_cot_samples + cot_samples
    n_instruct_samples = int(instruct_sample_proportion * len(total_task_samples))
    alpaca_samples = get_alpaca_training(n_instruct_samples)
    samples = (total_task_samples + alpaca_samples).shuffle("42")
    params = FineTuneParams(model=model, hyperparameters=FineTuneHyperParams(n_epochs=n_epochs))
    control_only_unbiased = formatter_options == FormatterOptions.control_only_unbiased
    more_config = {
        "instruct_sample_proportion": instruct_sample_proportion,
        "n_cots": len(cot_samples),
        "n_non_cots": len(non_cot_samples),
        "n_instruct_samples": len(alpaca_samples),
        "excluded_formatters": list(excluded_formatters_names),
        "eligible_non_cot_formatters": [sorted(eligible_non_cot_formatters.map(lambda x: x.name()))],
        "eligible_cot_formatters": [sorted(eligible_cot_formatters.map(lambda x: x.name()))],
        "formatter_options": formatter_options.value,
        "data_from": data_from_options.value,
        "post_hoc": post_hoc,
        "cot_percentage": cot_percentage,
        "control_only_unbiased": control_only_unbiased,
        "model_output_verified": model_output_verified.value,
    }
    cot_percentage_percentage = int(cot_percentage * 100)
    non_cot_percentage_percentage = int(non_cot_percentage * 100)
    bias_type_str = formatter_options.value + " bias formatters"
    notes = f"NO REPEATED FORMATTERS {bias_type_str} {cot_percentage_percentage}% cot {non_cot_percentage_percentage}% non cot, {n_samples} samples, {data_from_options.value} cots, {model_output_verified.value}"
    if post_hoc:
        notes = "post hoc " + notes
    _id = run_finetune_with_wandb(
        params=params,
        samples=samples,
        notes=notes,
        more_config=more_config,
        project_name=project_name,
        ask_to_validate_training=ask_to_validate_training,
    )
    return _id


class FormatSampler(ABC):
    @abstractmethod
    def sample(
        self,
        tasks: Sequence[TaskOutput],
        formatters: Sequence[FormatterWithPossibleIntervention],
        n: int,
    ) -> Slist[TaskOutput]:
        """
        Takes a sequnce of outputs and returns a sequence of outputs of length n
        """
        raise NotImplementedError


class NFormatsPerQuestionSampler(FormatSampler):
    def __init__(self, n_formats_per_question: int):
        self.n_formats_per_question = n_formats_per_question

    def sample(
        self,
        tasks: Sequence[TaskOutput],
        formatters: Sequence[FormatterWithPossibleIntervention],
        n: int,
    ) -> Slist[TaskOutput]:
        """
        Takes a sequnce of outputs and returns a sequence of outputs of length n
        """
        if self.n_formats_per_question > len(formatters):
            print("Warning: n_formats_per_question > len(formatters), using all formatters")

        n_formats_per_question = min(self.n_formats_per_question, len(formatters))

        tasks = Slist(tasks)
        n_unique_cots = math.ceil(n / n_formats_per_question)
        print("using n_unique_cots", n_unique_cots)
        tasks = tasks.take(n_unique_cots)

        output: Slist[TaskOutput] = Slist()
        formatter_counts = Counter()
        for task in tasks:
            rng = random.Random(task.uid())
            sampled_formatters = rng.sample(formatters, n_formats_per_question)
            formatter_counts.update(Counter([i.name() for i in sampled_formatters]))
            replaced = replace_unbiased_prompt_with_formatters(task=task, use_formatters=sampled_formatters)
            output.extend(replaced)

        output = output.take(n)
        assert len(output) == n
        print(f"Formatter counts:\n{json.dumps(formatter_counts, indent=2)}")

        return output

    def __repr__(self) -> str:
        return f"NFormatsPerQuestionSampler(n_formats_per_question={self.n_formats_per_question})"


class RandomSampler(FormatSampler):
    def sample(
        self,
        tasks: Sequence[TaskOutput],
        formatters: Sequence[FormatterWithPossibleIntervention],
        n: int,
    ) -> Slist[TaskOutput]:
        """
        Takes a sequnce of outputs and returns a sequence of outputs of length n
        """
        tasks = Slist(tasks)
        tasks = (
            tasks.map(lambda task: replace_unbiased_prompt_with_formatters(task=task, use_formatters=formatters))
            .flatten_list()
            .shuffle("42")
            .take(n)
        )
        assert len(tasks) == n, f"len(tasks)={len(tasks)}, n={n}"
        return tasks

    def __repr__(self) -> str:
        return "RandomSampler()"


def fine_tune_with_bias_augmentation(
    n_epochs: int,
    data_from_options: DataFromOptions = DataFromOptions.gpt_35_turbo,
    model_output_verified: ModelOutputVerified = ModelOutputVerified.correct,
    exclude_formatters: Sequence[type[StageOneFormatter]] = [],
    # if FormatterOptions.control_only_unbiased, then we only use unbiased contexts for training
    formatter_options: FormatterOptions = FormatterOptions.all_biased,
    project_name: str = "consistency-training",
    model: str = "gpt-3.5-turbo",
    n_samples: int = 72000,
    instruct_sample_proportion: float = 0.1,
    post_hoc: bool = False,
    cot_percentage=0.5,
    # cli waits for user input to validate the training
    ask_to_validate_training: bool = True,
    sampler: FormatSampler = RandomSampler(),
    prepend_notes: str = "",
    no_overlap_cot_non_cot: bool = False,
) -> str:
    """
    We use unbiased correct COTs, then replace the unbiased COT prompt with a biased COT formatter prompt
    """
    assert 0 <= cot_percentage <= 1
    assert 0 <= instruct_sample_proportion
    cot_limit = int(cot_percentage * n_samples)
    non_cot_percentage = 1 - cot_percentage
    non_cot_limit = int(non_cot_percentage * n_samples)
    excluded_formatters_names = {f.name() for f in exclude_formatters}
    match data_from_options:
        case DataFromOptions.gpt_35_turbo:
            non_cot_data = get_training_non_cots_gpt_35(model_output_verified)
            cot_data = get_training_cots_gpt_35(model_output_verified)
        case DataFromOptions.claude_2:
            non_cot_data = get_training_non_cots_claude_2(model_output_verified)
            cot_data = get_training_cots_claude_2(model_output_verified)
    non_cot_data_shuffled = non_cot_data.shuffle(seed="42")
    cot_data_shuffled = cot_data.shuffle(seed="42")
    formatter_options_result = match_formatter_options(formatter_options)
    non_cot_formatters = formatter_options_result.unbiased_formatters
    cot_formatters = formatter_options_result.biased_formatters

    eligible_non_cot_formatters = Slist(non_cot_formatters).filter(lambda x: x.formatter not in exclude_formatters)
    assert len(eligible_non_cot_formatters) > 0, "We do not have any eligible non cot formatters"
    eligible_cot_formatters = Slist(cot_formatters).filter(lambda x: x.formatter not in exclude_formatters)
    assert len(eligible_cot_formatters) > 0, "We do not have any eligible cot formatters"

    # Non Cots
    print(f"Number of non cots: {len(non_cot_data_shuffled)}")
    non_cot_tasks = sampler.sample(non_cot_data_shuffled, eligible_non_cot_formatters, non_cot_limit).map(
        augment_non_cot_task
    )

    non_cot_hashes: set[str] = {task.task_spec.task_hash for task in non_cot_tasks}
    print(f"Unique non cot hashes: {len(non_cot_hashes)}")

    non_cot_samples = non_cot_tasks.map(task_output_to_finetune_sample)

    assert (
        len(non_cot_samples) == non_cot_limit
    ), f"We do not have enough non cots, only {len(non_cot_samples)}, required {non_cot_limit}"
    print(f"Number of non cots after limiting: {len(non_cot_samples)}")

    # CoTs
    print(f"Number of cots: {len(cot_data_shuffled)}")
    cots_no_overlap = (
        cot_data_shuffled.filter(lambda task: task.task_spec.task_hash not in non_cot_hashes)
        if no_overlap_cot_non_cot
        else cot_data_shuffled
    )
    if no_overlap_cot_non_cot:
        print(f"Number of cots after removing overlap: {len(cots_no_overlap)}")
    cot_samples = (
        sampler.sample(cots_no_overlap, eligible_cot_formatters, cot_limit)
        .map(transform_into_post_hoc_reasoning if post_hoc else identity)
        .map(augment_cot_task)
        .map(task_output_to_finetune_sample)
    )

    assert len(cot_samples) == cot_limit, f"We do not have enough cots, only {len(cot_samples)}, required {cot_limit}"
    print(f"Number of cots after limiting: {len(cot_samples)}")

    total_task_samples = non_cot_samples + cot_samples
    n_instruct_samples = int(instruct_sample_proportion * len(total_task_samples))
    alpaca_samples = get_alpaca_training(n_instruct_samples)
    samples = (total_task_samples + alpaca_samples).shuffle("42")
    params = FineTuneParams(model=model, hyperparameters=FineTuneHyperParams(n_epochs=n_epochs))
    control_only_unbiased = formatter_options == FormatterOptions.control_only_unbiased

    more_config = {
        "instruct_sample_proportion": instruct_sample_proportion,
        "n_cots": len(cot_samples),
        "n_non_cots": len(non_cot_samples),
        "n_instruct_samples": len(alpaca_samples),
        "excluded_formatters": list(excluded_formatters_names),
        "eligible_non_cot_formatters": [sorted(eligible_non_cot_formatters.map(lambda x: x.name()))],
        "eligible_cot_formatters": [sorted(eligible_cot_formatters.map(lambda x: x.name()))],
        "formatter_options": formatter_options.value,
        "data_from": data_from_options.value,
        "post_hoc": post_hoc,
        "cot_percentage": cot_percentage,
        "control_only_unbiased": control_only_unbiased,
        "model_output_verified": model_output_verified.value,
        "sampling_strategy": sampler,
    }
    cot_percentage_percentage = int(cot_percentage * 100)
    non_cot_percentage_percentage = int(non_cot_percentage * 100)
    bias_type_str = formatter_options.value + " bias formatters"
    notes = f"{prepend_notes}{bias_type_str} {cot_percentage_percentage}% cot {non_cot_percentage_percentage}% non cot, {n_samples} samples, {data_from_options.value} cots, {model_output_verified.value}"
    if post_hoc:
        notes = "post hoc " + notes
    _id = run_finetune_with_wandb(
        params=params,
        samples=samples,
        notes=notes,
        more_config=more_config,
        project_name=project_name,
        ask_to_validate_training=ask_to_validate_training,
    )
    return _id


def fine_tune_with_dumb_brain_balanced(
    n_epochs: int,
    exclude_formatters: Sequence[type[StageOneFormatter]] = [],
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
    exclude_formatters: Sequence[type[StageOneFormatter]] = [],
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


def fine_tune_with_big_brain_cots_control_tokens(
    n: int,
    exclude_formattter: type[StageOneFormatter] | None,
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
    model = fine_tune_with_bias_augmentation(
        model="gpt-3.5-turbo",
        n_epochs=1,
        exclude_formatters=[
            WrongFewShotIgnoreMistakesBiasedFormatter,
            WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
        ],
        n_samples=10000,
        post_hoc=False,
        cot_percentage=0.50,
        project_name="consistency-training",
    )
