from dataclasses import dataclass
from enum import Enum

from scripts.finetune_cot import DataFromOptions, FormatSampler, FormatterOptions, RandomSampler


class FilterStrategy(str, Enum):
    no_filter = "no filter"
    correct_answer = "filtered to be correct"


@dataclass(frozen=True)
class ModelTrainMeta:
    name: str
    trained_samples: int
    filter_strategy: FilterStrategy
    sampling_strategy: FormatSampler = RandomSampler(formatter_options=FormatterOptions.zero_shot)
    data_from: DataFromOptions = DataFromOptions.gpt_35_turbo

    @property
    def train_formatters(self) -> str:
        return self.sampling_strategy.format_options_name

    def for_legend(self) -> str:
        return f"{self.sampling_strategy.format_options_name}, {self.filter_strategy.value}, {self.sampling_strategy.for_legend()}, {self.data_from.value}"
