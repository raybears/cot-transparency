from abc import ABC, abstractmethod
import re
from string import ascii_uppercase
from typing import Optional


from cot_transparency.data_models.example_base import (
    ChoiceVariant,
    DataFormatSpec,
)
from cot_transparency.data_models.example_base import IndicatorAndOption

BREAK_WORDS: list[str] = [
    "answer is (",
    "answer is  (",
    "answer is: (",
    "answer is:(",
    "answer is:  (",
    "answer is:\n(",
    "answer is: \n(",
    "answer is:\n\n(",
    "answer is: ",
    "answer is ",
    "answer is $\\boxed{\\text{(",
    "answer is: $\\boxed{\\text{(",
    "choices is: " r"is: $\boxed{\textbf{(",
    "answer: ",
    "answer is ",
    r"is: $\boxed{\textbf{(",
    "choices is: ",
    r"is: $\boxed{\textbf{(",
    r"is $\boxed{\textbf{(",
    r"is: $\boxed{\text{(",
]


class AnswerExtractor(ABC):
    @staticmethod
    @abstractmethod
    def extract(
        model_answer: str,
        options: list[str],
        dump_failed: bool = False,
        input_format: DataFormatSpec = DataFormatSpec(),
    ) -> Optional[str]:
        """
        Interface for answer extracting

        Args:
            model_answer (str): string response from the model
            options (list[str]): list of strings that are the literal answer options without indicators
            dump_failed (bool, optional): whether to dump failed examples to a text file. Defaults to False.
            input_format (DataFormatSpec, optional): format of the question. Defaults to DataFormatSpec().

        Returns:
            Optional[str]: returns string if found, None otherwise
        """

        raise NotImplementedError


class AnswerExtractorPipeline:
    def __init__(self, extractors: list[AnswerExtractor]):
        self.extractors = extractors

    def run_pipeline(
        self,
        model_answer: str,
        options: list[str],
        dump_failed: bool = False,
        input_format: DataFormatSpec = DataFormatSpec(),
    ) -> Optional[str]:
        for extractor in self.extractors:
            answer = extractor.extract(model_answer, options, dump_failed, input_format)
            if answer is not None:
                return answer
        return None


class FindIndicatorAfterBreakWord(AnswerExtractor):
    """
    Find answers in strings of the form "best answer is: (X)" and similar variants.
    """

    @staticmethod
    def extract(
        model_answer: str,
        options: list[str],
        dump_failed: bool = False,
        input_format: DataFormatSpec = DataFormatSpec(),
    ) -> Optional[str]:
        for break_word in BREAK_WORDS:
            if break_word not in model_answer:
                continue
            tmp = model_answer.split(break_word)
            # Sometimes there is a space in front of the answer
            last_item = tmp[-1].lstrip()

            if not last_item:
                continue

            # match single indicator or if it has a space, closing parenthesis, dot after it
            possible_indicators = input_format.choice_variant.answers_list[: len(options)]
            # also add lowercase variants
            possible_indicators_lower = [indicator.lower() for indicator in possible_indicators]
            possible_indicators_re = "|".join(possible_indicators + possible_indicators_lower)

            pattern = rf"^({possible_indicators_re})(\s|\)|\.|$)+.*$"

            match = re.search(pattern, last_item)
            if match:
                candidate_ans = match.group(1)
                if candidate_ans in possible_indicators:
                    idx = possible_indicators.index(candidate_ans)
                    return ascii_uppercase[idx]
                elif candidate_ans in possible_indicators_lower:
                    idx = possible_indicators_lower.index(candidate_ans)
                    return ascii_uppercase[idx]

        if dump_failed:
            with open("failed_answers.txt", "a") as f:
                f.write(model_answer + "\n")
        return None


class FindIndicatorAtStartOfResponse(AnswerExtractor):
    """
    Find answers in strings where the indicator comes at the start
    e.g. "A) answer or 2. answer"
    """

    @staticmethod
    def extract(
        model_answer: str,
        options: list[str],
        dump_failed: bool = False,
        input_format: DataFormatSpec = DataFormatSpec(),
    ) -> Optional[str]:
        # match single indicator or if it has a space, closing parenthesis, dot after it
        possible_indicators = input_format.choice_variant.answers_list[: len(options)]
        # also add lowercase variants
        possible_indicators_lower = [indicator.lower() for indicator in possible_indicators]
        possible_indicators_re = "|".join(possible_indicators + possible_indicators_lower)

        pattern = rf"^\(?({possible_indicators_re})(\s|\)|\.|$)+.*$"

        match = re.search(pattern, model_answer)
        if match:
            candidate_ans = match.group(1)
            if candidate_ans in possible_indicators:
                idx = possible_indicators.index(candidate_ans)
                return ascii_uppercase[idx]
            elif candidate_ans in possible_indicators_lower:
                idx = possible_indicators_lower.index(candidate_ans)
                return ascii_uppercase[idx]

        if dump_failed:
            with open("failed_answers.txt", "a") as f:
                f.write(model_answer + "\n")
        return None


def extract_answer():
    raise NotImplementedError("This is deprecated, use AnswerExtractorPipeline instead")


def extract_answer_looking_for_option():
    raise NotImplementedError("This is deprecated, use AnswerExtractorPipeline instead")


class FindAnswerStringAfterBreakWord(AnswerExtractor):
    """
    This Extractor finds literal answer strings after break words (does not have to be immediatley after)
    """

    @staticmethod
    def extract(
        model_answer: str,
        options: list[str],
        dump_failed: bool = False,
        input_format: DataFormatSpec = DataFormatSpec(),
    ) -> Optional[str]:
        for break_word in BREAK_WORDS:
            if break_word not in model_answer:
                continue
            tmp = model_answer.split(break_word)

            # see if the literal answer string is in the list
            for i, option in enumerate(options):
                # remove trailing periods - sometimes the model doesn't copy them.
                option_stripped = option.strip().rstrip(".")
                to_match_stripped = tmp[-1].strip().rstrip(".")
                if option_stripped in to_match_stripped:
                    return ascii_uppercase[i]

        if dump_failed:
            with open("failed_answers.txt", "a") as f:
                f.write(model_answer + "\n")
        return None


def extract_answer_non_cot(
    response: str,
    dump_failed: bool = False,
    input_format: DataFormatSpec = DataFormatSpec(),
    options: Optional[list[str]] = None,
) -> Optional[str]:
    ans_list = input_format.choice_variant.answers_list
    response = response.strip()

    # if options are provided, try to match the response to one of the options
    if options is not None:
        for i, option in enumerate(options):
            if option in response:
                return ascii_uppercase[i]

    pattern = re.compile(r"^\(?([a-zA-Z\d]+)\)?")

    match = pattern.match(response)
    if match:
        candidate_ans = match.group(1)

        # Convert to uppercase for consistent matching, except for the FOO variant
        if input_format.choice_variant != ChoiceVariant.FOO:
            candidate_ans = candidate_ans.upper()

        if candidate_ans in ans_list:
            idx = ans_list.index(candidate_ans)
            return ascii_uppercase[idx]

    # If we reach here, the answer failed to match the expected patterns
    if dump_failed:
        with open("failed_answers.txt", "a") as f:
            f.write(response + "\n")

    return None


def extract_multiple_choices(question: str) -> list[str]:
    """
    e.g.
    Q: Which of the following is a humorous edit of this artist or movie name: 'empire of the ants'?
    Answer choices:
    (A) empire of the pants
    (B) empiqe of the ants
    (C) empire of tihe ants
    (D) empire of the antts
    returns
    ['empire of the pants', 'empiqe of the ants', 'empire of tihe ants', 'empire of the antts']
    """
    # split the question into lines
    lines: list[str] = question.split("\n")
    # get index of the line that starts with "Answer choices:"
    index: int = lines.index("Answer choices:")
    # get the lines after that
    options: list[str] = lines[index + 1 :]
    # Get only the lines that start with a bracket
    options_with_bracket: list[str] = [option for option in options if option.startswith("(")]
    # Get only the text after the bracket
    options_without_bracket: list[str] = [option[option.index(")") + 1 :] for option in options_with_bracket]
    # strip
    stripped = [option.strip() for option in options_without_bracket]
    return stripped


def extract_lettered_multiple_choices(question: str) -> list[IndicatorAndOption]:
    extracted_answers = extract_multiple_choices(question)
    return [
        IndicatorAndOption(indicator=ascii_uppercase[i], option=answer) for i, answer in enumerate(extracted_answers)
    ]
