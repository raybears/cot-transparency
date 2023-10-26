import re
from abc import ABC, abstractmethod
from string import ascii_uppercase
from typing import Optional, Sequence

from fuzzywuzzy import fuzz

from cot_transparency.data_models.example_base import (
    ChoiceVariant,
    DataExampleBase,
    DataFormatSpec,
    IndicatorAndOption,
)

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
    r"answer is: \[\boxed{\text{",
    r"is: $\boxed{\textbf{(",
    "choices is: ",
    r"is: $\boxed{\textbf{(",
    r"is $\boxed{\textbf{(",
    r"is $\boxed{\text{(",
    r"is: \boxed{\text{(",
    r"is: $\boxed{\text{(",
    "accurate answer would be",
]


class AnswerExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        model_answer: str,
        dump_failed: bool = False,
    ) -> Optional[str]:
        """
        Interface for answer extracting

        Args:
            model_answer (str): string response from the model
            dump_failed (bool, optional): whether to dump failed examples to a text file. Defaults to False.

        Returns:
            Optional[str]: returns string if found, None otherwise
        """

        raise NotImplementedError


class AnswerExtractorPipeline:
    def __init__(self, extractors: Sequence[AnswerExtractor]):
        self.extractors = extractors

    def run_pipeline(
        self,
        model_answer: str,
        dump_failed: bool = False,
    ) -> Optional[str]:
        for extractor in self.extractors:
            answer = extractor.extract(model_answer, dump_failed)
            if answer is not None:
                return answer
        return None


class FindIndicatorAfterBreakWord(AnswerExtractor):
    """
    Find answers in strings of the form "best answer is: (X)" and similar variants.
    """

    def __init__(
        self,
        options: list[str],
        input_format: DataFormatSpec = DataFormatSpec(),
        break_words: list[str] = BREAK_WORDS,
    ):
        """
        Args:
            options (list[str]): list of strings that are the literal answer options without indicators
            input_format (DataFormatSpec, optional): format of the question. Defaults to DataFormatSpec().
        """

        self.options = options
        self.input_format = input_format
        self.break_words = break_words

    def extract(
        self,
        model_answer: str,
        dump_failed: bool = False,
    ) -> Optional[str]:
        for break_word in self.break_words:
            if break_word not in model_answer:
                continue
            tmp = model_answer.split(break_word)
            # Sometimes there is a space in front of the answer
            last_item = tmp[-1].lstrip()

            if not last_item:
                continue

            # match single indicator or if it has a space, closing parenthesis, dot after it
            possible_indicators = self.input_format.choice_variant.answers_list[: len(self.options)]
            # also add lowercase variants
            possible_indicators_lower = [indicator.lower() for indicator in possible_indicators]
            possible_indicators_re = "|".join(possible_indicators + possible_indicators_lower)

            pattern = rf"^({possible_indicators_re})(\s|\)|\.|$)+.*$"
            pattern = rf"^(?:[Oo]ption |[Ss]tatement )?\(?({possible_indicators_re})\)?(\s|\)|\.|$)+.*$"

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
    e.g. "A) answer or 2. answer or (B) answer"
    """

    def __init__(self, options: list[str], input_format: DataFormatSpec = DataFormatSpec()):
        """
        Args:
            options (list[str]): list of strings that are the literal answer options without indicators
            input_format (DataFormatSpec, optional): format of the question. Defaults to DataFormatSpec().
        """
        self.options = options
        self.input_format = input_format

    def extract(
        self,
        model_answer: str,
        dump_failed: bool = False,
    ) -> Optional[str]:
        # match single indicator or if it has a space, closing parenthesis, dot after it
        possible_indicators = self.input_format.choice_variant.answers_list[: len(self.options)]
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


class FindAnswerStringAfterBreakWord(AnswerExtractor):
    """
    This Extractor finds literal answer strings after break words (does not have to be immediatley after)
    """

    def __init__(self, options: list[str]):
        """
        Args:
            options (list[str]): list of strings that are the literal answer options without indicators

        """
        self.options = options

    def extract(
        self,
        model_answer: str,
        dump_failed: bool = False,
    ) -> Optional[str]:
        for break_word in BREAK_WORDS:
            if break_word not in model_answer:
                continue
            tmp = model_answer.split(break_word)

            # see if the literal answer string is in the list
            for i, option in enumerate(self.options):
                # remove trailing periods - sometimes the model doesn't copy them.
                option_stripped = option.strip().rstrip(".").lower()
                to_match_stripped = tmp[-1].strip().rstrip(".").lower()
                if option_stripped in to_match_stripped:
                    return ascii_uppercase[i]

        if dump_failed:
            with open("failed_answers.txt", "a") as f:
                f.write(model_answer + "\n")
        return None


class FuzzyMatcher(AnswerExtractor):
    def __init__(self, options: list[str], match_threshold: int = 84):
        self.options = options
        self.match_threshold = match_threshold

    def extract(
        self,
        model_answer: str,
        dump_failed: bool = False,
    ) -> Optional[str]:
        if "answer is" in model_answer:
            model_answer = model_answer.split("answer is")[1]
        else:
            return None

        # fuzzy match model_answer with options using fuzzywuzzy and choose the best match
        fuzz_scores = []
        for option in self.options:
            fuzz_scores.append(fuzz.ratio(model_answer, option))  # type: ignore

        # if the best match is above a certain threshold, return the answer
        if max(fuzz_scores) > self.match_threshold:
            return ascii_uppercase[fuzz_scores.index(max(fuzz_scores))]  # type: ignore


def extract_answer(response: str, question: DataExampleBase, dump_failed: bool = False) -> Optional[str]:
    """
    This is a legacy method to find answers that use letters, recomended to use AnswerExtractorPipeline directly instead
    """

    options = question.get_options()
    input_format = question.data_format
    extractors = [
        FindIndicatorAfterBreakWord(options, input_format),
    ]
    return AnswerExtractorPipeline(extractors).run_pipeline(response, dump_failed)


def extract_answer_looking_for_option():
    raise NotImplementedError("This is deprecated, use AnswerExtractorPipeline instead")


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
