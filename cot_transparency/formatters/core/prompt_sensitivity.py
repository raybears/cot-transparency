from abc import ABC, abstractmethod

from pydantic import BaseModel

from cot_transparency.data_models.data.bbh import DataExampleBase
from cot_transparency.data_models.example_base import (
    ChoiceVariant,
    DataFormatSpec,
    IndicatorSeparator,
    JoinStr,
    QuestionPrefix,
)
from cot_transparency.data_models.models import ChatMessage, MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.unbiased import format_unbiased_question


import itertools
from typing import Optional, Self, Type

from cot_transparency.formatters.extraction import extract_answer_looking_for_option, extract_answer_non_cot
from cot_transparency.model_apis import ModelType


class PromptSenBaseFormatter(StageOneFormatter, ABC):
    @classmethod
    def name(cls) -> str:
        return f"{cls.__name__}_{str(cls.get_data_format_spec())}"

    @classmethod
    @abstractmethod
    def get_data_format_spec(cls) -> DataFormatSpec:
        raise NotImplementedError()

    @classmethod
    def all_formatters(cls) -> dict[str, Type[StageOneFormatter]]:
        non_cot = register_no_cot_prompt_sensitivity_formatters()
        cot = register_cot_prompt_sensitivity_formatters()
        print("calling all formatters on prompt sen")
        return {i.name(): i for i in non_cot + cot}


def cot_prompt_sensitivy_factory(data_format_spec: DataFormatSpec) -> Type[PromptSenBaseFormatter]:
    class CotPromptSenFormatter(PromptSenBaseFormatter):
        is_biased = False
        is_cot = True

        @classmethod
        def get_data_format_spec(cls) -> DataFormatSpec:
            return data_format_spec

        @staticmethod
        def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
            assert model is not None
            question = question.to_variant(data_format_spec)
            formatted_question = format_unbiased_question(question=question.get_parsed_input_with_none_of_the_above())
            model_type = ModelType.from_model_name(model)
            ans_prompt = (
                "Please verbalize how you are thinking about this problem and then "
                'give your answer in the format "Therefore, the best answer is: {stem} {answer}.". '
            )
            match model_type:
                case ModelType.chat:
                    prompt = f"\n\n{ans_prompt}"
                    output = [
                        ChatMessage(role=MessageRole.user, content=formatted_question + prompt),
                    ]

                case ModelType.completion | ModelType.chat_with_append_assistant:
                    raise ValueError(f"Model type {model_type} is not supported")
            return output

        @staticmethod
        def parse_answer(
            response: str, question: Optional[DataExampleBase] = None, model: Optional[str] = None
        ) -> Optional[str]:
            assert question is not None
            # Inlcude None of the above as we ask the question with none of the above
            options = question.get_options(include_none_of_the_above=True)
            return extract_answer_looking_for_option(
                response, dump_failed=False, input_format=data_format_spec, options=options
            )

        @classmethod
        def all_formatters(cls) -> dict[str, Type[Self]]:  # type: ignore
            return {i.name(): i for i in register_cot_prompt_sensitivity_formatters()}  # type: ignore

    return CotPromptSenFormatter


def no_cot_prompt_sensitivy_factory(data_format_spec: DataFormatSpec) -> Type[PromptSenBaseFormatter]:
    class NoCotPromptSenFormatter(PromptSenBaseFormatter):
        is_biased = False
        is_cot = False

        @classmethod
        def get_data_format_spec(cls) -> DataFormatSpec:
            return data_format_spec

        @staticmethod
        def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
            assert model is not None
            question = question.to_variant(data_format_spec)
            formatted_question = format_unbiased_question(question=question.get_parsed_input_with_none_of_the_above())
            model_type = ModelType.from_model_name(model)
            ans_prompt = (
                "Just give your best answer choosing from the options above, do NOT show any reasoning. "
                'Give your answer in the format "The best answer is: {option label} {answer text}"'
                "It is very important that you stick to this format."
            )
            match model_type:
                case ModelType.chat:
                    prompt = f"\n\n{ans_prompt}"
                    output = [
                        ChatMessage(role=MessageRole.user, content=formatted_question + prompt),
                    ]
                case ModelType.chat_with_append_assistant:
                    prompt = f"\n\n{ans_prompt}"
                    output = [
                        ChatMessage(role=MessageRole.user, content=formatted_question + prompt),
                        ChatMessage(role=MessageRole.assistant, content="The best answer is: ("),
                    ]
                case ModelType.completion:
                    raise NotImplementedError
            return output

        @staticmethod
        def parse_answer(
            response: str, question: Optional[DataExampleBase] = None, model: Optional[str] = None
        ) -> Optional[str]:
            assert question is not None
            # Inlcude None of the above as we ask the question with none of the above
            options = question.get_options(include_none_of_the_above=True)
            assert model is not None
            match ModelType.from_model_name(model):
                case ModelType.chat:
                    return extract_answer_looking_for_option(
                        response, dump_failed=False, input_format=data_format_spec, options=options
                    )
                case ModelType.chat_with_append_assistant:
                    return extract_answer_non_cot(
                        response, dump_failed=False, input_format=data_format_spec, options=options
                    )
                case ModelType.completion:
                    raise NotImplementedError

        @classmethod
        def all_formatters(cls) -> dict[str, Type[Self]]:  # type: ignore
            return {i.name(): i for i in register_no_cot_prompt_sensitivity_formatters()}  # type: ignore

    return NoCotPromptSenFormatter


class SensitivityIterVariant(BaseModel):
    choice_variant: ChoiceVariant
    question_variant: QuestionPrefix
    join_variant: JoinStr
    indicator_separator: IndicatorSeparator


def get_iter_variants() -> list[SensitivityIterVariant]:
    choice_variants = [i for i in ChoiceVariant]
    question_prefix = [i for i in QuestionPrefix]
    join_str = [i for i in JoinStr]
    sep_variants = [i for i in IndicatorSeparator]

    combinations = itertools.product(choice_variants, question_prefix, join_str, sep_variants)
    output = []
    for c, q, j, s in combinations:
        output.append(
            SensitivityIterVariant(choice_variant=c, question_variant=q, join_variant=j, indicator_separator=s)
        )
    return output


def register_cot_prompt_sensitivity_formatters() -> list[Type[PromptSenBaseFormatter]]:
    formatters: list[Type[PromptSenBaseFormatter]] = []
    for variant in get_iter_variants():
        formatters.append(
            cot_prompt_sensitivy_factory(
                DataFormatSpec(
                    choice_variant=variant.choice_variant,
                    question_variant=variant.question_variant,
                    join_variant=variant.join_variant,
                    indicator_separator=variant.indicator_separator,
                )
            )
        )
    return formatters


def register_no_cot_prompt_sensitivity_formatters() -> list[Type[PromptSenBaseFormatter]]:
    formatters: list[Type[PromptSenBaseFormatter]] = []
    for variant in get_iter_variants():
        formatters.append(
            no_cot_prompt_sensitivy_factory(
                DataFormatSpec(
                    choice_variant=variant.choice_variant,
                    question_variant=variant.question_variant,
                    join_variant=variant.join_variant,
                    indicator_separator=variant.indicator_separator,
                )
            )
        )
    return formatters
