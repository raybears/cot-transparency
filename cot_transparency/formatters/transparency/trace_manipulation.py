import re

from typing import List
import nltk


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
    # hack to preserve newline characters as punkt tokenizer removes them
    # but we want split on new lines separately
    line = line.replace("\n", "<NEWLINE>")

    # split on sentence boundaries while preserving whitespace
    # using punkt tokenizer
    segments = nltk.tokenize.sent_tokenize(line)
    # add back in the whitespace that was removed by the tokenizer
    segments_with_whitespace = []
    for i, segment in enumerate(segments):
        if i > 0:
            segments_with_whitespace.append(f" {segment}")
        else:
            segments_with_whitespace.append(segment)

    # remove empty strings
    segments = [segment for segment in segments_with_whitespace if segment != ""]

    # add back in the newline characters
    segments = [segment.replace("<NEWLINE>", "\n") for segment in segments]
    return segments


def split_string(text: str) -> List[str]:
    lines = split_on_newlines(text)
    # split each line on sentences
    sentences = [sentence for line in lines for sentence in split_on_sentences(line)]
    return sentences


def get_cot_steps(cot_part: str) -> List[str]:
    # Split on sentence and line breaks as per "Measuring Transparency in Chain-of-Thought Reasoning"
    cot_sentences = split_string(cot_part)

    cot_steps = [""]
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
