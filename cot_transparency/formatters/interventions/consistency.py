from typing import Optional, Type

from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import ChatMessage
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.interventions.few_shots_loading import (
    get_correct_cots,
    get_big_brain_cots,
    get_correct_cots_claude_2,
)
from cot_transparency.formatters.interventions.intervention import (
    Intervention,
)
from cot_transparency.formatters.interventions.formatting import (
    format_pair_cot,
    format_unbiased_question_cot,
    format_biased_question_cot,
    prepend_to_front_first_user_message,
    format_few_shot_for_prompt_sen,
    format_biased_question_non_cot_random_formatter,
    format_pair_non_cot,
    format_biased_question_non_cot_sycophancy,
    format_big_brain_question_cot,
    insert_to_after_system_message,
)
from cot_transparency.model_apis import Prompt


class PairedConsistency6(Intervention):
    # Because it is a pair, sample 6 / 2 = 3
    n_samples: int = 3

    @classmethod
    def intervene(
        cls, question: DataExampleBase, formatter: Type[StageOneFormatter], model: Optional[str] = None
    ) -> list[ChatMessage]:
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


class RepeatedConsistency10(Intervention):
    # Just the naive few shot, but repeated 5 * 2 = 10
    n_samples: int = 5

    @classmethod
    def intervene(
        cls, question: DataExampleBase, formatter: Type[StageOneFormatter], model: Optional[str] = None
    ) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        cots = get_correct_cots().sample(cls.n_samples, seed=question.hash())
        duplicated = cots + cots
        prompt: Prompt = duplicated.map(format_unbiased_question_cot).sum_or_raise()
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class RepeatedConsistency12(RepeatedConsistency10):
    n_samples: int = 6


class BiasedConsistency10(Intervention):
    @classmethod
    def intervene(
        cls, question: DataExampleBase, formatter: Type[StageOneFormatter], model: Optional[str] = None
    ) -> list[ChatMessage]:
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


class BigBrainBiasedConsistency10(Intervention):
    n_samples: int = 10

    @classmethod
    def intervene(
        cls, question: DataExampleBase, formatter: Type[StageOneFormatter], model: Optional[str] = None
    ) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        # TODO: filter out to not sample the same biased formatter
        prompt: Prompt = (
            # Not a pair so, sample 10
            get_big_brain_cots()
            .sample(cls.n_samples, seed=question.hash())
            .map(format_big_brain_question_cot)
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class BigBrainBiasedConsistency12(BigBrainBiasedConsistency10):
    n_samples: int = 12


class BigBrainBiasedConsistencySeparate10(BigBrainBiasedConsistency10):
    """Separate the few shots into messages rather than in the single message"""

    @classmethod
    def intervene(
        cls, question: DataExampleBase, formatter: Type[StageOneFormatter], model: Optional[str] = None
    ) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            # Not a pair so, sample 10
            get_big_brain_cots()
            .sample(10, seed=question.hash())
            .map(format_big_brain_question_cot)
            .sum_or_raise()
        )
        new = insert_to_after_system_message(messages=messages, to_insert=prompt.messages)
        return new


class NaiveFewShot1(Intervention):
    # Simply use unbiased few shot
    n_samples: int = 1

    @classmethod
    def intervene(
        cls, question: DataExampleBase, formatter: Type[StageOneFormatter], model: Optional[str] = None
    ) -> list[ChatMessage]:
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


class NaiveFewShot5(NaiveFewShot1):
    # Simply use unbiased few shot
    n_samples: int = 5


class NaiveFewShot10(NaiveFewShot1):
    # Simply use unbiased few shot
    n_samples: int = 10


class NaiveFewShot12(NaiveFewShot1):
    # Simply use unbiased few shot
    n_samples: int = 12


class NaiveFewShot16(NaiveFewShot1):
    # Simply use unbiased few shot
    n_samples: int = 16


class NaiveFewShot32(NaiveFewShot1):
    # Simply use unbiased few shot
    n_samples: int = 32


class ClaudeFewShot1(Intervention):
    n_samples: int = 1

    @classmethod
    def intervene(
        cls, question: DataExampleBase, formatter: Type[StageOneFormatter], model: Optional[str] = None
    ) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots_claude_2()
            .sample(cls.n_samples, seed=question.hash())
            .map(format_unbiased_question_cot)
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class ClaudeFewShot3(ClaudeFewShot1):
    n_samples: int = 3


class ClaudeFewShot6(ClaudeFewShot1):
    n_samples: int = 6


class ClaudeFewShot10(ClaudeFewShot1):
    n_samples: int = 10


class ClaudeSeparate10(Intervention):
    # Simply use unbiased few shot
    n_samples: int = 10

    @classmethod
    def intervene(
        cls, question: DataExampleBase, formatter: Type[StageOneFormatter], model: Optional[str] = None
    ) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots_claude_2()
            .sample(cls.n_samples, seed=question.hash())
            .map(format_unbiased_question_cot)
            .sum_or_raise()
        )
        new = insert_to_after_system_message(
            messages=messages,
            to_insert=prompt.messages,
        )
        return new


class ClaudeFewShot16(ClaudeFewShot1):
    n_samples: int = 16


class ClaudeFewShot32(ClaudeFewShot1):
    n_samples: int = 16


class NaiveFewShotSeparate10(Intervention):
    # Simply use unbiased few shot
    n_samples: int = 10

    @classmethod
    def intervene(
        cls, question: DataExampleBase, formatter: Type[StageOneFormatter], model: Optional[str] = None
    ) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots()
            .sample(cls.n_samples, seed=question.hash())
            .map(format_unbiased_question_cot)
            .sum_or_raise()
        )
        new = insert_to_after_system_message(
            messages=messages,
            to_insert=prompt.messages,
        )
        return new


class NaiveFewShotLabelOnly1(Intervention):
    # Non cot, only the label
    n_samples: int = 1

    @classmethod
    def intervene(
        cls, question: DataExampleBase, formatter: Type[StageOneFormatter], model: Optional[str] = None
    ) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots()
            .sample(cls.n_samples, seed=question.hash())
            .map(format_few_shot_for_prompt_sen)
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
    def intervene(
        cls, question: DataExampleBase, formatter: Type[StageOneFormatter], model: Optional[str] = None
    ) -> list[ChatMessage]:
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
    def intervene(
        cls, question: DataExampleBase, formatter: Type[StageOneFormatter], model: Optional[str] = None
    ) -> list[ChatMessage]:
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
