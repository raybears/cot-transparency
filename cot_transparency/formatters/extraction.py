import re
from string import ascii_uppercase
from typing import Optional


from cot_transparency.data_models.example_base import (
    ChoiceVariant,
    DataFormatSpec,
    combine_indicator_with_separator,
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
    r"is $\boxed{\textbf{(",
    r"is: $\boxed{\text{(",
]


def extract_answer(model_answer: str, dump_failed: bool = False) -> Optional[str]:
    """
    Find answers in strings of the form "best answer is: (X)" and similar variants.
    """

    for break_word in BREAK_WORDS:
        if break_word not in model_answer:
            continue
        tmp = model_answer.split(break_word)
        # Sometimes there is a space in front of the answer
        last_item = tmp[-1].lstrip()
        ans = last_item[0]
        if ans in ascii_uppercase:
            return ans
    if dump_failed:
        with open("failed_answers.txt", "a") as f:
            f.write(model_answer + "\n")
    return None


def extract_answer_looking_for_option(
    model_answer: str, dump_failed=False, input_format=DataFormatSpec(), options: Optional[list[str]] = None
) -> Optional[str]:
    for break_word in BREAK_WORDS:
        if break_word not in model_answer:
            continue
        tmp = model_answer.split(break_word)

        # first see if the literal answer string is in the list
        if options is not None:
            for i, option in enumerate(options):
                if option in tmp[-1].strip():
                    return ascii_uppercase[i]

        ans_list = input_format.choice_variant.answers_list
        combined = [combine_indicator_with_separator(ans, input_format.indicator_separator).strip() for ans in ans_list]

        for i, ans_indicator in enumerate(combined):
            if ans_indicator in tmp[-1]:
                idx = combined.index(ans_indicator)
                return ascii_uppercase[idx]

        # also allow matching on the answer without separators
        parsed_ans_without_separator = tmp[-1].replace(")", "").replace(".", "").strip()
        for ans_indicator in ans_list:
            if ans_indicator == parsed_ans_without_separator:
                idx = ans_list.index(ans_indicator)
                return ascii_uppercase[idx]

        for ans in ans_list:
            # match if the ans is in a \textbf{()} box with optional parentheses
            pattern = r"\\text(bf)?{{\s*\({}\)?\}}".format(ans)
            match = re.search(pattern, tmp[-1])
            if match:
                idx = ans_list.index(ans)
                return ascii_uppercase[idx]

    if dump_failed:
        with open("failed_answers.txt", "a") as f:
            f.write(model_answer + "\n")
    return None


def extract_answer_non_cot(
    response: str, dump_failed: bool = False, input_format: DataFormatSpec = DataFormatSpec()
) -> Optional[str]:
    ans_list = input_format.choice_variant.answers_list
    response = response.strip()

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
