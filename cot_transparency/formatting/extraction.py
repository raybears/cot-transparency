from string import ascii_uppercase
from typing import Optional


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


def extract_answer_non_cot(model_answer: str, dump_failed: bool = False) -> Optional[str]:
    if model_answer[0] in ascii_uppercase:
        return model_answer[0]
    if dump_failed:
        with open("failed_answers.txt", "a") as f:
            f.write(model_answer + "\n")
    return None
