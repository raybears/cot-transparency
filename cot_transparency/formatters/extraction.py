from string import ascii_uppercase
from typing import Optional

from cot_transparency.data_models.example_base import VALID_ANSWERS, MultipleChoiceAnswer


def extract_answer(model_answer: str, dump_failed: bool = False) -> Optional[str]:
    # This is kinda janky lol
    break_words = [
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
    for break_word in break_words:
        if break_word not in model_answer:
            continue
        tmp = model_answer.split(break_word)
        ans = tmp[-1][0]
        if ans in ascii_uppercase:
            return ans
        break
    if dump_failed:
        with open("failed_answers.txt", "a") as f:
            f.write(model_answer + "\n")
    return None


def extract_answer_non_cot(model_answer: str, dump_failed: bool = False) -> Optional[MultipleChoiceAnswer]:
    if model_answer[0] in VALID_ANSWERS:
        return model_answer[0]
    if len(model_answer) > 1 and model_answer[1] in VALID_ANSWERS:
        return model_answer[1]
    if dump_failed:
        with open("failed_answers.txt", "a") as f:
            f.write(model_answer + "\n")
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
