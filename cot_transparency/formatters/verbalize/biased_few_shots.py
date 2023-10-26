import re
from enum import Enum
from typing import Match, Optional

from pydantic import BaseModel


class ReturnedBiasFailure(str, Enum):
    no_bias_detected = "no_bias_detected"
    # model didn't say NO_BIAS_DETECTED but also didn't return bias
    failed = "failed"


class BiasAndExplanation(BaseModel):
    bias: ReturnedBiasFailure | str
    explanation: ReturnedBiasFailure | str
    raw_response: Optional[str] = None


def parse_out_bias_explanation(completion: str) -> BiasAndExplanation:
    # the bias is wrapped in <BIAS>bias name</BIAS>
    bias_parsed: Match[str] | None = re.search(r"<BIAS>(.*)</BIAS>", completion)
    explicit_no_bias_detected: bool = "NO_BIAS_DETECTED" in completion
    bias: str | ReturnedBiasFailure = (
        bias_parsed.group(1).strip()
        if bias_parsed
        else ReturnedBiasFailure.no_bias_detected
        if explicit_no_bias_detected
        else ReturnedBiasFailure.failed
    )
    explanation: Match[str] | None = re.search(
        r"<EXPLANATION>(.*)</EXPLANATION>", completion
    )
    return BiasAndExplanation(
        bias=bias,
        explanation=explanation.group(1).strip()
        if explanation
        else ReturnedBiasFailure.failed,
        raw_response=completion,
    )
