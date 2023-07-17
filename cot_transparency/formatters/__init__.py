from typing import Type
from cot_transparency.formatters.sycophancy import (
    ZeroShotCOTSycophancyFormatter,
    ZeroShotCOTSycophancyNoRoleFormatter,
    ZeroShotCOTSycophancyToldBiasFormatter,
    ZeroShotSycophancyFormatter,
    ZeroShotSycophancyNoRoleFormatter,
)
from cot_transparency.formatters.unbiased import (
    ZeroShotCOTUnbiasedFormatter,
    ZeroShotCOTUnbiasedNoRoleFormatter,
    ZeroShotUnbiasedFormatter,
    ZeroShotUnbiasedNoRoleFormatter,
)
from cot_transparency.formatters.base_class import PromptFormatter


def bias_to_unbiased_formatter(biased_formatter_name: str) -> str:
    mapping = {
        ZeroShotCOTSycophancyFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        ZeroShotSycophancyFormatter.name(): ZeroShotUnbiasedFormatter.name(),
        ZeroShotSycophancyNoRoleFormatter.name(): ZeroShotUnbiasedNoRoleFormatter.name(),
        ZeroShotCOTSycophancyNoRoleFormatter.name(): ZeroShotCOTUnbiasedNoRoleFormatter.name(),
        ZeroShotCOTSycophancyToldBiasFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
    }
    return mapping[biased_formatter_name]


def name_to_formatter(name: str) -> Type[PromptFormatter]:
    mapping = PromptFormatter.all_formatters()
    return mapping[name]
