import re
from string import ascii_uppercase
from typing import Optional

from pydantic import BaseModel

from cot_transparency.data_models.example_base import ChoiceVariant, MultipleChoiceAnswer

BREAK_WORDS: list[str] = [
    "best answer is (",
    "best answer is  (",
    "best answer is: (",
    "best answer is:(",
    "best answer is:  (",
    "best answer is:\n(",
    "best answer is: \n(",
    "best answer is:\n\n(",
    "best answer is: ",
    "best answer is ",
    "answer is (",
    "answer is: (",
    "answer is $\\boxed{\\text{(",
    "answer is: $\\boxed{\\text{(",
    "is: $\\boxed{\\textbf{(",
    "is $\\boxed{\\textbf{(",
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
        break
    if dump_failed:
        with open("failed_answers.txt", "a") as f:
            f.write(model_answer + "\n")
    return None


def extract_answer_non_cot(
    response: str, dump_failed: bool = False, input_format: ChoiceVariant = ChoiceVariant.LETTERS
) -> Optional[str]:
    ans_list = input_format.answers_list
    response = response.strip()

    # Define a pattern that captures a string that starts with a
    # "(" or a letter/number, followed by characters, and optionally followed by a ")".
    pattern = re.compile(r"^\(?([a-zA-Z\d]+)\)?")

    match = pattern.match(response)
    if match:
        candidate_ans = match.group(1)

        # Convert to uppercase for consistent matching, except for the FOO variant
        if input_format != ChoiceVariant.FOO:
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


class LetterAndAnswer(BaseModel):
    letter: MultipleChoiceAnswer
    answer: str


def extract_lettered_multiple_choices(question: str) -> list[LetterAndAnswer]:
    extracted_answers = extract_multiple_choices(question)
    return [
        LetterAndAnswer(letter=ascii_uppercase[i], answer=answer)  # type: ignore
        for i, answer in enumerate(extracted_answers)
    ]
