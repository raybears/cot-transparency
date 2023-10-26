from typing import Type

from slist import Slist

from cot_transparency.data_models.example_base import DataFormatSpec
from cot_transparency.formatters.prompt_sensitivity.v1_prompt_sen import (
    PromptSenBaseFormatter,
    register_cot_prompt_sensitivity_formatters,
    register_no_cot_prompt_sensitivity_formatters,
)

no_cot_sensitivity_formatters: list[
    Type[PromptSenBaseFormatter]
] = register_no_cot_prompt_sensitivity_formatters()
cot_sensitivity_formatters: list[
    Type[PromptSenBaseFormatter]
] = register_cot_prompt_sensitivity_formatters()

# Retrieve the formatter from the DataFormatSpec using these maps
cot_sensitivity_data_format_spec_map: dict[
    DataFormatSpec, Type[PromptSenBaseFormatter]
] = (
    Slist(cot_sensitivity_formatters)
    .group_by(lambda x: x.get_data_format_spec())  # type: ignore
    .map_2(lambda key, values: (key, values.first_or_raise()))
    .to_dict()
)
no_cot_sensitivity_data_format_spec_map: dict[
    DataFormatSpec, Type[PromptSenBaseFormatter]
] = (
    Slist(no_cot_sensitivity_formatters)
    .group_by(lambda x: x.get_data_format_spec())  # type: ignore
    .map_2(lambda key, values: (key, values.first_or_raise()))
    .to_dict()
)
default_no_cot_sensitivity_formatter = no_cot_sensitivity_data_format_spec_map[
    DataFormatSpec()
]
