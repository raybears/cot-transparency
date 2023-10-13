import fnmatch
from typing import Sequence

from cot_transparency.formatters import StageOneFormatter


def match_wildcard_formatters(formatters: Sequence[str]) -> list[str]:
    new_formatters = list(formatters)
    for formatter in new_formatters:
        if "*" in formatter:
            new_formatters.remove(formatter)
            new_formatters += fnmatch.filter(StageOneFormatter.all_formatters().keys(), formatter)
    return new_formatters
