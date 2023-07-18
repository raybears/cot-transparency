import re

from typing import List


def split_text_preserve_case(text: str, separator: str) -> List[str]:
    parts = None

    if separator.lower() in text:
        parts = text.split(separator.lower(), maxsplit=1)
        parts[1] = separator.lower() + parts[1]
    elif separator in text:
        parts = text.split(separator, maxsplit=1)
        parts[1] = separator + parts[1]
    else:
        raise ValueError(f"Separator '{separator}' not found in text")
    return parts


def get_blank_cot_inp(inp: str, cot_part: str, tokenizer, ans_part=None) -> str:
    num_reasoning_tokens = len(tokenizer.encode(cot_part))
    blank_cot = " ".join(["..." for _ in range(num_reasoning_tokens)])
    out = f"{inp}\n\n" + blank_cot
    if ans_part:
        out = out + " " + ans_part
    return out


def contains_reasoning(step: str) -> bool:
    """Returns True if the sentence is a valid reasoning step, False otherwise.
    Examples of invalid reasoning steps: '1.', only new line characters, only spaces
    """
    stripped = step.strip()
    if not stripped:
        return False
    if re.match(r"^\d+.?$", stripped):
        return False
    return True


def split_on_newlines(text: str) -> List[str]:
    # split on newlines while preserving newline characters
    segments = re.split("(?=\\n)", text)
    # remove empty strings
    segments = [segment for segment in segments if segment != ""]
    return segments


def split_on_sentences(line: str) -> List[str]:
    # split on sentence boundaries while preserving whitespace
    segments = re.split("(?<=[.!?])(?=\\s)", line)
    # remove empty strings
    segments = [segment for segment in segments if segment != ""]
    return segments


def split_string(text: str) -> List[str]:
    lines = split_on_newlines(text)
    # split each line on sentences
    sentences = [sentence for line in lines for sentence in split_on_sentences(line)]
    return sentences


def get_cot_steps(cot_part: str) -> List[str]:
    # Split on sentence and line breaks as per "Measuring Transparency in Chain-of-Thought Reasoning"
    cot_sentences = split_string(cot_part)

    cot_steps = []
    prefix = ""
    for step in cot_sentences:
        if not contains_reasoning(step):
            # if the step only contains spaces or new lines, don't count it
            # but include it to preserve the formatting of the original output
            prefix += step
        else:
            cot_steps.append(prefix + step)
            prefix = ""
    return cot_steps
