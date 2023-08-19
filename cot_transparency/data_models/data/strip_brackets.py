import re


def extract_answer_from_brackets(string: str) -> str:
    # e.g. " (A)" -> "A"
    # e.g. "A" -> "A"
    # e.g. "(A)" -> "A"
    return re.sub(r"[()]", "", string).strip()
