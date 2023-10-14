from typing import Type

from cot_transparency.formatters.base_class import StageOneFormatter, PromptFormatter


def name_to_formatter(name: str) -> Type[PromptFormatter]:
    if name in _MAPPING_STORE:
        return _MAPPING_STORE[name]
    else:
        mapping = PromptFormatter.all_formatters()
        _MAPPING_STORE.update(mapping)
        return mapping[name]


_MAPPING_STORE: dict[str, Type[PromptFormatter]] = {}
_STAGE_ONE_MAPPING: dict[str, Type[StageOneFormatter]] = {}


def name_to_stage1_formatter(name: str) -> Type[StageOneFormatter]:
    if name in _STAGE_ONE_MAPPING:
        return _STAGE_ONE_MAPPING[name]
    else:
        mapping = StageOneFormatter.all_formatters()
        _STAGE_ONE_MAPPING.update(mapping)
        return mapping[name]
