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


def split_on_sentences(line: str, preserve_new_lines: bool = True) -> List[str]:
    # hack to preserve newline characters as punkt tokenizer removes them
    # but we want split on new lines separately
    if preserve_new_lines:
        line = line.replace("\n", "<NEWLINE>")
    else:
        line = line.replace("\n", " ")

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

    # if not preserving new lines ensure segments all end with a full stop
    if not preserve_new_lines:
        for i, segment in enumerate(segments[1:]):
            if not segment.endswith("."):
                segment = segment + "."
                segments[i + 1] = segment

    return segments


def split_string(text: str, preserve_line_breaks: bool = True) -> List[str]:
    lines = split_on_newlines(text)
    # split each line on sentences
    sentences = [
        sentence
        for line in lines
        for sentence in split_on_sentences(
            line,
            preserve_new_lines=preserve_line_breaks,
        )
    ]
    return sentences


def get_cot_steps(cot_part: str, preserve_line_breaks: bool = True) -> List[str]:
    # Split on sentence and line breaks as per "Measuring Transparency in Chain-of-Thought Reasoning"
    cot_sentences = split_string(cot_part, preserve_line_breaks=preserve_line_breaks)

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

    if prefix != "":
        # ensure we aleays return everything we've been given even when it's not a valid reasoning step
        cot_steps.append(prefix)
    return cot_steps
