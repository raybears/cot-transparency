from string import ascii_uppercase
from typing import Optional

from pydantic import BaseModel

from cot_transparency.data_models.example_base import MultipleChoiceAnswer

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
]


def extract_answer(model_answer: str, dump_failed: bool = False) -> Optional[str]:
    # This is kinda janky lol

    for break_word in BREAK_WORDS:
        if break_word not in model_answer:
            continue
        tmp = model_answer.split(break_word)
        second_item = tmp[1] if len(tmp) >= 2 else None
        assert second_item is not None
        ans = second_item[0]
        if ans in ascii_uppercase:
            return ans
        break
    if dump_failed:
        with open("failed_answers.txt", "a") as f:
            f.write(model_answer + "\n")
    return None


def extract_answer_non_cot(
    response: str, allow_failure: bool = True, dump_failed: bool = False
) -> Optional[MultipleChoiceAnswer]:
    out: Optional[MultipleChoiceAnswer] = None
    response = response.strip()
    if len(response) >= 1:
        if response[0].upper() in ascii_uppercase:
            out = response[0]  # type: ignore
            if len(response) > 1:
                if response[1] != ")":
                    out = None
        if out is None and allow_failure:
            print(f"Did not find a valid answer in response '{response}', but allow_failure is set to True")
            return None

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


class LetterAndAnswer(BaseModel):
    letter: MultipleChoiceAnswer
    answer: str


def extract_lettered_multiple_choices(question: str) -> list[LetterAndAnswer]:
    extracted_answers = extract_multiple_choices(question)
    return [
        LetterAndAnswer(letter=ascii_uppercase[i], answer=answer)  # type: ignore
        for i, answer in enumerate(extracted_answers)
    ]
