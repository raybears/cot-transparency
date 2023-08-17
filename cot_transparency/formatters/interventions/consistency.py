from typing import Type

from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import ChatMessage
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.interventions.few_shots_loading import get_correct_cots
from cot_transparency.formatters.interventions.intervention import (
    Intervention,
)
from cot_transparency.formatters.interventions.formatting import (
    format_pair_cot,
    format_unbiased_question_cot,
    format_biased_question_cot,
    prepend_to_front_first_user_message,
    format_unbiased_question_non_cot,
    format_biased_question_non_cot_random_formatter,
    format_pair_non_cot,
    format_biased_question_non_cot_sycophancy,
)
from cot_transparency.model_apis import Prompt


class PairedConsistency6(Intervention):
    # Because it is a pair, sample 6 / 2 = 3
    n_samples: int = 3

    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots().sample(cls.n_samples, seed=question.hash()).map(format_pair_cot).sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class PairedConsistency12(PairedConsistency6):
    # Because it is a pair, sample 12 / 2 = 6
    n_samples: int = 6


class PairedConsistency10(PairedConsistency6):
    # Because it is a pair, sample 10 / 2 = 5
    n_samples: int = 5


class BiasedConsistency10(Intervention):
    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            # Not a pair so, sample 10
            get_correct_cots()
            .sample(10, seed=question.hash())
            .map(lambda task: format_biased_question_cot(task=task, formatter=formatter))
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class NaiveFewShot1(Intervention):
    # Simply use unbiased few shot
    n_samples: int = 1

    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots()
            .sample(cls.n_samples, seed=question.hash())
            .map(format_unbiased_question_cot)
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class NaiveFewShot3(NaiveFewShot1):
    # Simply use unbiased few shot
    n_samples: int = 3


class NaiveFewShot6(NaiveFewShot1):
    # Simply use unbiased few shot
    n_samples: int = 6


class NaiveFewShot10(NaiveFewShot1):
    # Simply use unbiased few shot
    n_samples: int = 10


class NaiveFewShot12(NaiveFewShot1):
    # Simply use unbiased few shot
    n_samples: int = 12


class NaiveFewShot16(NaiveFewShot1):
    # Simply use unbiased few shot
    n_samples: int = 16


class NaiveFewShotLabelOnly1(Intervention):
    # Non cot, only the label
    n_samples: int = 1

    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots()
            .sample(cls.n_samples, seed=question.hash())
            .map(format_unbiased_question_non_cot)
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class NaiveFewShotLabelOnly3(NaiveFewShotLabelOnly1):
    n_samples: int = 3


class NaiveFewShotLabelOnly6(NaiveFewShotLabelOnly1):
    n_samples: int = 6


class NaiveFewShotLabelOnly10(Intervention):
    n_samples: int = 10


class NaiveFewShotLabelOnly16(NaiveFewShotLabelOnly1):
    n_samples: int = 16


class NaiveFewShotLabelOnly30(NaiveFewShotLabelOnly1):
    n_samples: int = 30


class NaiveFewShotLabelOnly32(NaiveFewShotLabelOnly30):
    n_samples: int = 32


class SycophancyConsistencyLabelOnly10(Intervention):
    n_samples: int = 10

    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots()
            .sample(cls.n_samples, seed=question.hash())
            .map(format_biased_question_non_cot_sycophancy)
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class SycoConsistencyLabelOnly30(SycophancyConsistencyLabelOnly10):
    n_samples: int = 30


class BiasedConsistencyLabelOnly10(Intervention):
    n_samples: int = 10

    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots()
            .sample(10, seed=question.hash())
            .map(lambda task: format_biased_question_non_cot_random_formatter(task=task, formatter=formatter))
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class BiasedConsistencyLabelOnly20(BiasedConsistencyLabelOnly10):
    n_samples: int = 20


class BiasedConsistencyLabelOnly30(BiasedConsistencyLabelOnly10):
    n_samples: int = 30


class PairedFewShotLabelOnly10(Intervention):
    # Non cot, only the label
    # Because it is a pair, sample 10 / 2 = 5
    n_samples: int = 5

    @classmethod
    def hook(cls, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        prompt: Prompt = (
            get_correct_cots().sample(cls.n_samples, seed=question.hash()).map(format_pair_non_cot).sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class PairedFewShotLabelOnly30(PairedFewShotLabelOnly10):
    # Non cot, only the label
    # Because it is a pair, sample 30 / 2 = 15
    n_samples: int = 15
