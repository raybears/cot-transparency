import dataclasses
import json
import math
import random
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Sequence
from enum import Enum
from pathlib import Path

from slist import Slist, identity

from cot_transparency.apis.openai.finetune import (
    FineTuneHyperParams,
    FineTuneParams,
    FinetuneSample,
    run_finetune_with_wandb,
)
from cot_transparency.copy_utils.unset_sentinel import Unset, _UNSET
from cot_transparency.data_models.data.gpt_35_instructions import get_all_alpaca_training_gpt_35
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.example_base import DummyDataExample
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.data_models.models import BaseTaskOutput
from cot_transparency.data_models.streaming import ParaphrasingOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.unbiased import (
    ZeroShotCOTUnbiasedFormatter,
    ZeroShotUnbiasedFormatter,
)
from cot_transparency.formatters.instructions import VERBALIZE_INSTRUCTION
from cot_transparency.formatters.interventions.few_shots_loading import (
    ModelOutputVerified,
    get_training_cots_claude_2,
    get_training_cots_gpt_35,
    get_training_cots_gpt_35_gs,
    get_training_non_cots_claude_2,
    get_training_non_cots_gpt_35,
    get_training_non_cots_gpt_35_gs,
    task_output_to_finetune_sample,
)
from cot_transparency.formatters.interventions.intervention import Intervention
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotIgnoreMistakesBiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
)
from cot_transparency.formatters.name_mapping import name_to_formatter
from cot_transparency.formatters.prompt_sensitivity.automated_generations import (
    GoldStandardNoCotFormatter,
    GoldStandardWithCotFormatter,
    GoldStandardWithCotFormatter2,
    GoldStandardWithCotFormatter3,
    GoldStandardWithCotFormatter4,
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
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel
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
    def augment(messages: Sequence[ChatMessage]) -> Sequence[ChatMessage]:
        return list(messages)


class RandomCOTPromptAugmentor(Augmentor):
    @staticmethod
    def augment(messages: Sequence[ChatMessage]) -> Sequence[ChatMessage]:
        new = []
        for message in messages:
            content: str = message.content
            if VERBALIZE_INSTRUCTION in content:
                content = content.replace(VERBALIZE_INSTRUCTION, sample_cot_variant(content))
            new.append(ChatMessage(role=message.role, content=content))
        return new


class RandomNonCOTPromptAugmentor(Augmentor):
    @staticmethod
    def augment(messages: Sequence[ChatMessage]) -> Sequence[ChatMessage]:
        messages = Slist(messages)
        # ref to the first user message
        first_user_idx: int = messages.find_one_idx_or_raise(lambda x: x.role == MessageRole.user)
        content = messages[first_user_idx].content
        # edit the first user message
        sampled_no_cot_instruction: str = content + "\n" + non_sample_cot_variant(seed=content)
        messages[first_user_idx] = ChatMessage(role=MessageRole.user, content=sampled_no_cot_instruction)

        return messages


def augment_non_cot_task(item: BaseTaskOutput) -> BaseTaskOutput:
    new_messages = RandomNonCOTPromptAugmentor.augment(messages=item.get_task_spec().messages)
    return item.update_messages_in_task_spec(messages=new_messages)


def augment_cot_task(item: BaseTaskOutput) -> BaseTaskOutput:
    new_messages = RandomCOTPromptAugmentor.augment(messages=item.get_task_spec().messages)
    return item.update_messages_in_task_spec(messages=new_messages)


def replace_unbiased_cot_prompt_with_formatters(
    task: BaseTaskOutput,
    use_formatters: Iterable[type[StageOneFormatter]],
    intervention: type[Intervention] | None = None,
) -> Slist[BaseTaskOutput]:
    output = Slist[BaseTaskOutput]()
    for formatter in use_formatters:
        new = task.model_copy(deep=True)

        assert (
            task.get_task_spec().formatter_name == ZeroShotCOTUnbiasedFormatter.name()
        ), f"Got {task.get_task_spec().formatter_name}"
        data_example: DataExampleBase = task.get_task_spec().get_data_example_obj()
        if intervention is not None:
            new.get_task_spec().messages = intervention.intervene(question=data_example, formatter=formatter)
        else:
            new.get_task_spec().messages = formatter.format_example(data_example)
        output.append(new)
    return output


def transform_into_post_hoc_reasoning(task: BaseTaskOutput) -> BaseTaskOutput:
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
    task: BaseTaskOutput,
    use_formatters: Iterable[FormatterWithPossibleIntervention],
) -> Slist[BaseTaskOutput]:
    output = Slist[BaseTaskOutput]()
    for fwpi in use_formatters:
        new = task
        data_example: DataExampleBase = task.get_task_spec().get_data_example_obj()
        intervention = fwpi.intervention
        new_messages = (
            fwpi.formatter.format_example(data_example)
            if intervention is None
            else intervention.intervene(
                question=data_example,
                formatter=fwpi.formatter,
                model=task.get_task_spec().inference_config.model,
            )
        )
        new = new.update_messages_in_task_spec(new_messages)
        output.append(new)
    return output


class DataFromOptions(str, Enum):
    gpt_35_turbo = "gpt-3.5-turbo"
    claude_2 = "claude-2"
    # gold standard formatter, doesn't specify  that the model should answer with "The best answer is: "
    gpt_35_turbo_gs = "gpt-3.5-turbo-gs"
    gpt_35_turbo_gs2 = "gpt-3.5-turbo-gs2"
    gpt_35_turbo_gs3 = "gpt-3.5-turbo-gs3"
    gpt_35_turbo_gs4 = "gpt-3.5-turbo-gs4"


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
    ask_paraphrased = "ask_paraphrased"
    gs_unbiased = "gs_unbiased"


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
        case FormatterOptions.ask_paraphrased:
            non_cot_formatters = [FormatterWithPossibleIntervention(formatter=ZeroShotUnbiasedFormatter)]
            cot_formatters = [FormatterWithPossibleIntervention(formatter=ZeroShotCOTUnbiasedFormatter)]

        case FormatterOptions.gs_unbiased:
            cot_formatters = [FormatterWithPossibleIntervention(formatter=ZeroShotUnbiasedFormatter)]
            non_cot_formatters = [FormatterWithPossibleIntervention(formatter=ZeroShotCOTUnbiasedFormatter)]

    return FormatterOptionsResult(
        biased_formatters=sorted(list(set(cot_formatters))),
        unbiased_formatters=sorted(list(set(non_cot_formatters))),
    )


class FormatSampler(ABC):
    @abstractmethod
    def sample(
        self,
        tasks: Sequence[BaseTaskOutput],
        formatters: Sequence[FormatterWithPossibleIntervention],
        n: int,
    ) -> Slist[BaseTaskOutput]:
        """
        Takes a sequnce of outputs and returns a sequence of outputs of length n
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def for_legend(self) -> str:
        raise NotImplementedError


class NFormatsPerQuestionSampler(FormatSampler):
    def __init__(self, n_formats_per_question: int):
        self.n_formats_per_question = n_formats_per_question

    def sample(
        self,
        tasks: Sequence[BaseTaskOutput],
        formatters: Sequence[FormatterWithPossibleIntervention],
        n: int,
    ) -> Slist[BaseTaskOutput]:
        """
        Takes a sequnce of outputs and returns a sequence of outputs of length n
        """
        if self.n_formats_per_question > len(formatters):
            print(
                f"Warning: n_formats_per_question={self.n_formats_per_question} > len(formatters):{len(formatters)}: , using all formatters"
            )

        n_formats_per_question = min(self.n_formats_per_question, len(formatters))

        tasks = Slist(tasks)
        n_unique_cots = math.ceil(n / n_formats_per_question)
        print("using n_unique_cots", n_unique_cots)
        tasks = tasks.take(n_unique_cots)

        output: Slist[BaseTaskOutput] = Slist()
        formatter_counts = Counter()
        for task in tasks:
            rng = random.Random(task.uid())
            sampled_formatters = rng.sample(formatters, n_formats_per_question)
            formatter_counts.update(Counter([i.name() for i in sampled_formatters]))
            replaced = replace_unbiased_prompt_with_formatters(task=task, use_formatters=sampled_formatters)
            output.extend(replaced)

        output = output.take(n)
        assert len(output) == n, f"len(output)={len(output)}, n={n}"
        print(f"Formatter counts:\n{json.dumps(formatter_counts, indent=2)}")

        return output

    def __repr__(self) -> str:
        return f"NFormatsPerQuestionSampler(n_formats_per_question={self.n_formats_per_question})"

    def for_legend(self) -> str:
        return f"n_formats={self.n_formats_per_question}"


class RandomSampler(FormatSampler):
    def sample(
        self,
        tasks: Sequence[BaseTaskOutput],
        formatters: Sequence[FormatterWithPossibleIntervention],
        n: int,
    ) -> Slist[BaseTaskOutput]:
        """
        Takes a sequence of outputs and returns a sequence of outputs of length n
        """
        tasks = Slist(tasks)
        tasks = (
            tasks.map(lambda task: replace_unbiased_prompt_with_formatters(task=task, use_formatters=formatters))
            .flatten_list()
            # IMPORTANT: we shuffle the tasks before taking n
            # Otherwise, we will always have a small amount of unique questions
            .shuffle(seed="42")
            .take(n)
        )
        assert len(tasks) == n, f"len(tasks)={len(tasks)}, n={n}"
        return tasks

    def __repr__(self) -> str:
        return "RandomSampler()"

    def for_legend(self) -> str:
        return "Random Sampling"


class ParaphrasingSampler(FormatSampler):
    """
    This is a sort of dummy sampler so that we can get the paraphrased questions
    """

    def __init__(
        self,
        n_formats_per_question: int,
        use_unique_cots: bool = False,
        cot_paraphrasings_file="data/training_paraphrasings/GenerateParaphrasingsFormatters.jsonl",
        non_cot_paraphrasings_file="data/training_paraphrasings/GenerateParaphrasingsNoCotFormatters.jsonl",
    ):
        # load the paraphrasings
        self.n_formats_per_question = n_formats_per_question
        self.use_unique_cots = use_unique_cots

        self.cot_paraphrasings_file = cot_paraphrasings_file
        self.non_cot_paraphrasings_file = non_cot_paraphrasings_file

        self.cot_mapping: dict[str, ParaphrasingOutput] = {}
        self.no_cot_mapping: dict[str, ParaphrasingOutput] = {}

    def load_paraphrasings(self):
        for is_cot in [True, False]:
            if is_cot:
                paraphrasings = read_jsonl_file_into_basemodel(self.cot_paraphrasings_file, ParaphrasingOutput)
            else:
                paraphrasings = read_jsonl_file_into_basemodel(self.non_cot_paraphrasings_file, ParaphrasingOutput)

            for paraphrasing in paraphrasings:
                key = self._get_key(paraphrasing)
                if is_cot:
                    self.cot_mapping[key] = paraphrasing
                else:
                    self.no_cot_mapping[key] = paraphrasing

    def _get_key(self, task: BaseTaskOutput) -> str:
        return (
            task.get_task_spec().get_task_hash()
            + "isCot="
            + str(name_to_formatter(task.get_task_spec().formatter_name).is_cot)
        )

    def sample(
        self,
        tasks: Sequence[BaseTaskOutput],
        formatters: Sequence[FormatterWithPossibleIntervention],
        n: int,
    ) -> Slist[BaseTaskOutput]:
        tasks = Slist(tasks)

        if not self.cot_mapping:
            self.load_paraphrasings()

        supported_formatters = [ZeroShotUnbiasedFormatter, ZeroShotCOTUnbiasedFormatter]

        for f in Slist(formatters).map(lambda x: x.formatter):
            assert f in supported_formatters, f"Formatter {f} not supported for paraphrasing sampler"

        group = tasks.group_by(lambda x: x.get_task_spec().get_task_hash())
        if self.use_unique_cots:
            assert group.all(lambda x: len(x.values) >= self.use_unique_cots)

        ret = Slist()
        for task_group in group:
            if self.use_unique_cots:
                tasks = task_group.values.take(self.n_formats_per_question)
            else:
                tasks = task_group.values.take(1).repeat_until_size_or_raise(self.n_formats_per_question)
            key = self._get_key(tasks[0])
            for f in formatters:
                mapping = self.cot_mapping if f.formatter.is_cot else self.no_cot_mapping
                paraphrasing = mapping[key]
                paraphrased_questions = Slist(paraphrasing.paraphrased_questions)
                to_use = paraphrased_questions.shuffle(seed=key).take(self.n_formats_per_question)

                for task, paraphrased in zip(tasks, to_use):
                    paraphrased_as_data_example = DummyDataExample(parsed_input=paraphrased.paraphrased)

                    intervention = f.intervention
                    if intervention is not None:
                        new_messages = intervention.intervene(
                            question=task.get_task_spec().get_data_example_obj(),
                            formatter=f.formatter,
                            model=task.get_task_spec().inference_config.model,
                        )
                    else:
                        new_messages = f.formatter.format_example(paraphrased_as_data_example)

                    new_task = task.update_messages_in_task_spec(messages=new_messages)
                    ret.append(new_task)

        ret = ret.take(n)
        if len(ret) < n:
            breakpoint()
        assert len(ret) == n, "Not enough paraphrased questions"
        return ret

    def __repr__(self) -> str:
        if not self.use_unique_cots:
            return f"ParaphrasingSampler(n_formats_per_question={self.n_formats_per_question})"
        else:
            return f"ParaphrasingSampler(n_formats_per_question={self.n_formats_per_question}, use_unique_cots={self.use_unique_cots})"

    def for_legend(self) -> str:
        return f"n_formats={self.n_formats_per_question}"


class InstructSource(str, Enum):
    # Completions from the alpaca dataset
    alpaca_original = "alpaca_original"
    # Completions from the alpaca dataset, but with gpt-3.5-turbo-0613 completions
    alpaca_gpt_35 = "alpaca_gpt_35"


class CombinedSampler(FormatSampler):
    def __init__(self, samplers: Sequence[FormatSampler]):
        self.samplers = Slist(samplers)

    def sample(
        self,
        tasks: Sequence[BaseTaskOutput],
        formatters: Sequence[FormatterWithPossibleIntervention],
        n: int,
    ) -> Slist[BaseTaskOutput]:
        tasks = Slist(tasks)
        # split the tasks into n groups
        groups = tasks.split_into_n(len(self.samplers))
        # sample from each group
        ret = Slist()
        for sampler, group in zip(self.samplers, groups):
            ret.extend(sampler.sample(group, formatters, math.ceil(n / len(self.samplers))))
        return ret.take(n)

    def __repr__(self) -> str:
        return f"CombinedSampler(samplers={self.samplers})"

    def for_legend(self) -> str:
        return "Combined"


def fine_tune_with_bias_augmentation(
    # TODO: deprecate
    n_epochs: int | Unset = _UNSET,
    hyperparams: FineTuneHyperParams = FineTuneHyperParams(n_epochs=1),
    data_from_options: DataFromOptions = DataFromOptions.gpt_35_turbo,
    model_output_verified: ModelOutputVerified = ModelOutputVerified.unfiltered,
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
    # For now we recommend using NFormatsPerQuestionSampler=1, rather than RandomSampler
    sampler: FormatSampler = NFormatsPerQuestionSampler(n_formats_per_question=1),
    val_sampler: FormatSampler = NFormatsPerQuestionSampler(n_formats_per_question=1),
    prepend_notes: str = "",
    # If true, we permute the verbalize instructions to have multiple variations
    permute_verbalize_instructions: bool = True,
    # Ensures that the cot and non cot questions do not overlap
    # This is useful so the chance of overlaps between the cot and non cot questions does not
    # change when we change the size of the training data
    no_overlap_cot_non_cot: bool = True,
    n_val_samples: int = 1000,
    # Choose InstructSource.alpaca_gpt_35 if you want to use the gpt-3.5-turbo-0613 completions
    instruct_source: InstructSource = InstructSource.alpaca_gpt_35,
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
        case DataFromOptions.gpt_35_turbo_gs:
            non_cot_data = get_training_non_cots_gpt_35_gs(model_output_verified, GoldStandardNoCotFormatter)
            cot_data = get_training_cots_gpt_35_gs(model_output_verified, GoldStandardWithCotFormatter)
        case DataFromOptions.gpt_35_turbo_gs2:
            non_cot_data = get_training_non_cots_gpt_35_gs(model_output_verified, GoldStandardNoCotFormatter)
            cot_data = get_training_cots_gpt_35_gs(model_output_verified, GoldStandardWithCotFormatter2)
        case DataFromOptions.gpt_35_turbo_gs3:
            non_cot_data = get_training_non_cots_gpt_35_gs(model_output_verified, GoldStandardNoCotFormatter)
            cot_data = get_training_cots_gpt_35_gs(model_output_verified, GoldStandardWithCotFormatter3)
        case DataFromOptions.gpt_35_turbo_gs4:
            non_cot_data = get_training_non_cots_gpt_35_gs(model_output_verified, GoldStandardNoCotFormatter)
            cot_data = get_training_cots_gpt_35_gs(model_output_verified, GoldStandardWithCotFormatter4)

    non_cot_data_shuffled = Slist(non_cot_data).shuffle(seed="42")
    # use a different seed for cots and non cots in case the data is in the same order
    cot_data_shuffled = Slist(cot_data).shuffle(seed="1")
    formatter_options_result = match_formatter_options(formatter_options)
    non_cot_formatters = formatter_options_result.unbiased_formatters
    cot_formatters = formatter_options_result.biased_formatters

    eligible_non_cot_formatters = Slist(non_cot_formatters).filter(lambda x: x.formatter not in exclude_formatters)
    # assert len(eligible_non_cot_formatters) > 0, "We do not have any eligible non cot formatters"
    eligible_cot_formatters = Slist(cot_formatters).filter(lambda x: x.formatter not in exclude_formatters)
    # assert len(eligible_cot_formatters) > 0, "We do not have any eligible cot formatters"

    # Non Cots
    print(f"Number of non cots: {len(non_cot_data_shuffled)}")

    # split of val samples
    n_non_cot_val_samples = int(n_val_samples * (1 - cot_percentage))
    n_cot_val_samples = int(n_val_samples * cot_percentage)

    val_task_hashes: set[str] = set()
    non_cot_data_val = non_cot_data_shuffled.take(n_non_cot_val_samples + 20)
    non_cot_data_val.for_each(lambda x: val_task_hashes.add(x.get_task_spec().get_task_hash()))
    cot_data_val = cot_data_shuffled.filter(lambda x: x.get_task_spec().get_task_hash() not in val_task_hashes).take(
        n_cot_val_samples + 20
    )

    non_cot_tasks = non_cot_data_shuffled.filter(lambda x: x.get_task_spec().get_task_hash() not in val_task_hashes)
    # remove the val samples from the non cot data
    non_cot_samples, non_cot_hashes = get_non_cot_samples(
        non_cot_tasks,
        eligible_non_cot_formatters,
        non_cot_limit,
        sampler,
        permute_verbalize_instructions,
    )

    print(f"Unique non cot hashes: {len(non_cot_hashes)}")

    print(f"Number of non cots after limiting: {len(non_cot_samples)}")
    non_cot_val_samples, _ = get_non_cot_samples(
        non_cot_data_val,
        eligible_non_cot_formatters,
        n_non_cot_val_samples,
        val_sampler,
        permute_verbalize_instructions,
    )
    print(f"Number of validation non cots after limiting: {len(non_cot_val_samples)}")

    # CoTs
    print(f"Number of cots: {len(cot_data_shuffled)}")
    # Make sure cot_samples doesn't contain any of the val samples
    # And if no_overlap_cot_non_cot, make sure cot_samples doesn't contain any of the non_cot_samples
    val_and_non_cot_hashes = val_task_hashes.union(non_cot_hashes) if no_overlap_cot_non_cot else val_task_hashes
    cot_tasks = cot_data_shuffled.filter(lambda x: x.get_task_spec().get_task_hash() not in val_and_non_cot_hashes)
    cot_samples, cot_hashes = get_cot_samples(
        cot_tasks,
        eligible_cot_formatters,
        cot_limit,
        sampler,
        post_hoc,
        permute_verbalize_instructions,
    )
    if no_overlap_cot_non_cot:
        print(f"Number of cots after removing overlap: {len(cot_samples)}")

    assert len(cot_samples) == cot_limit, f"We do not have enough cots, only {len(cot_samples)}, required {cot_limit}"
    print(f"Number of cots after limiting: {len(cot_samples)}")
    cot_val_samples, _ = get_cot_samples(
        cot_data_val,
        eligible_cot_formatters,
        n_cot_val_samples,
        val_sampler,
        post_hoc,
        permute_verbalize_instructions,
    )
    print(f"Number of validation cots after limiting: {len(cot_val_samples)}")

    if no_overlap_cot_non_cot:
        assert non_cot_hashes.isdisjoint(cot_hashes), "cot and non cot hashes are not disjoint, this is a bug"

    total_task_samples = non_cot_samples + cot_samples
    val_instruct_samples = int(n_val_samples * instruct_sample_proportion)
    n_instruct_samples = int(instruct_sample_proportion * len(total_task_samples))

    match instruct_source:
        case InstructSource.alpaca_original:
            alpaca_samples = get_alpaca_training(n_instruct_samples + val_instruct_samples)
        case InstructSource.alpaca_gpt_35:
            alpaca_samples = get_all_alpaca_training_gpt_35(seed="42", limit=n_instruct_samples + val_instruct_samples)
    alpaca_train_samples, alpaca_val_samples = (
        alpaca_samples[:-val_instruct_samples],
        alpaca_samples[-val_instruct_samples:],
    )

    samples = (total_task_samples + alpaca_train_samples).shuffle("42")
    val_samples = (non_cot_val_samples + cot_val_samples + alpaca_val_samples).shuffle("42")

    if n_epochs is not _UNSET:
        print("WARNING: n_epochs is deprecated, use hyperparams instead, setting n_epochs to hyperparams.n_epochs")
        hyperparams = hyperparams.model_copy(deep=True, update={"n_epochs": n_epochs})
    params = FineTuneParams(model=model, hyperparameters=hyperparams)
    control_only_unbiased = formatter_options == FormatterOptions.control_only_unbiased

    more_config = {
        "instruct_sample_proportion": instruct_sample_proportion,
        "n_cots": len(cot_samples),
        "n_non_cots": len(non_cot_samples),
        "n_unique_cot_questions": len(cot_hashes),
        "n_unique_non_cot_questions": len(non_cot_hashes),
        "n_instruct_samples": len(alpaca_samples),
        "n_val_cots": len(cot_val_samples),
        "n_val_non_cots": len(non_cot_val_samples),
        "n_val_samples": len(val_samples),
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
        "permute_verbalize_instructions": permute_verbalize_instructions,
        "no_overlap_cot_non_cot": no_overlap_cot_non_cot,
        "instructions_source": instruct_source.value,
        "cot_paraphrasings_from": sampler.cot_paraphrasings_file if isinstance(sampler, ParaphrasingSampler) else None,
        "non_cot_paraphrasings_from": sampler.non_cot_paraphrasings_file
        if isinstance(sampler, ParaphrasingSampler)
        else None,
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
        val_samples=val_samples,
    )
    return _id


def get_non_cot_samples(
    shuffled_data: Sequence[BaseTaskOutput],
    eligible_formatters: Slist[FormatterWithPossibleIntervention],
    limit: int,
    sampler: FormatSampler,
    permute_verbalize_instructions: bool,
) -> tuple[Slist[FinetuneSample], set[str]]:
    non_cot_samples = sampler.sample(shuffled_data, eligible_formatters, limit).map(
        lambda x: augment_non_cot_task(x) if permute_verbalize_instructions else x
    )
    hashes = non_cot_samples.map(lambda x: x.get_task_spec().get_task_hash()).to_set()
    non_cot_samples = non_cot_samples.map(task_output_to_finetune_sample)

    assert (
        len(non_cot_samples) == limit
    ), f"We do not have enough non cots, only {len(non_cot_samples)}, required {limit}"
    return non_cot_samples, hashes


def get_cot_samples(
    shuffled_data: Sequence[BaseTaskOutput],
    eligible_cot_formatters: Slist[FormatterWithPossibleIntervention],
    limit: int,
    sampler: FormatSampler,
    post_hoc: bool,
    permute_verbalize_instructions: bool,
) -> tuple[Slist[FinetuneSample], set[str]]:
    cot_samples = (
        sampler.sample(shuffled_data, eligible_cot_formatters, limit)
        .map(transform_into_post_hoc_reasoning if post_hoc else identity)
        .map(lambda x: augment_cot_task(x) if permute_verbalize_instructions else x)
    )
    hashes = cot_samples.map(lambda x: x.get_task_spec().get_task_hash()).to_set()
    cot_samples = cot_samples.map(task_output_to_finetune_sample)

    assert len(cot_samples) == limit, f"We do not have enough cots, only {len(cot_samples)}"
    return cot_samples, hashes


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
