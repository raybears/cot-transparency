from string import ascii_uppercase
from typing import Optional

from cot_transparency.data_models.example_base import MultipleChoiceAnswer, LetterAndOption

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
    r"is: $\boxed{\textbf{(",
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


def extract_answer_non_cot(response: str, dump_failed: bool = False) -> Optional[MultipleChoiceAnswer]:
    out: Optional[MultipleChoiceAnswer] = None
    response = response.strip()
    if len(response) >= 1:
        if response[0].upper() in ascii_uppercase:
            out = response[0]  # type: ignore
            if len(response) > 1:
                if response[1] != ")":
                    out = None

        elif response[0] == "(" and response[1].upper() in ascii_uppercase:
            out = response[1]  # type: ignore

        if out is None and dump_failed:
            with open("failed_answers.txt", "a") as f:
                f.write(response + "\n")

    return out


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


def extract_lettered_multiple_choices(question: str) -> list[LetterAndOption]:
    extracted_answers = extract_multiple_choices(question)
    return [
        LetterAndOption(letter=ascii_uppercase[i], option=answer)  # type: ignore
        for i, answer in enumerate(extracted_answers)
    ]
