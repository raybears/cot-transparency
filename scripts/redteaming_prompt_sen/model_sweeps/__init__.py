from enum import Enum

from slist import Slist

from scripts.finetune_cot import FormatterOptions, NFormatsPerQuestionSampler
from scripts.finetune_zero_shot_experiments.comparison_plot import (
    FilterStrategy,
    ModelTrainMeta,
)
from scripts.redteaming_prompt_sen.model_sweeps.biases import FEW_SHOT, FEW_SHOT_2, ZERO_SHOT_2
from scripts.redteaming_prompt_sen.model_sweeps.biases import ZERO_SHOT
from scripts.redteaming_prompt_sen.model_sweeps.biases import OG_CONTROL
from scripts.redteaming_prompt_sen.model_sweeps.paraphrasing import PARAPHRASING_1
from scripts.redteaming_prompt_sen.model_sweeps.prompt_variants import PROMPT_VARIANT_1, PROMPT_VARIANTS_ALL_2


N_FORMATS = [
    # prompt variant models, trained with 4 formats per question
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8Cc8jA11",
        trained_samples=20000,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.prompt_variants_set1,
        sampling_strategy=NFormatsPerQuestionSampler(4),
    ),
    # Hail Mary - all formats
    # ft:gpt-3.5-turbo-0613:academicsnyuperez::8CbgrYvU
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8CbgrYvU",
        trained_samples=20000,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.prompt_variants_all,
        sampling_strategy=NFormatsPerQuestionSampler(4),
    ),
]


GPT = [
    ModelTrainMeta(
        name="gpt-3.5-turbo",
        trained_samples=1,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.control_only_unbiased,
        sampling_strategy=NFormatsPerQuestionSampler(1),
    ),
]


class Sweeps(str, Enum):
    paraphrasing_1 = "paraphrasing_1"
    paraphrasing_2 = "paraphrasing_2"
    paraphrasing_2_correct = "paraphrasing_2_correct"
    paraphrasing_5 = "paraphrasing_5"
    prompt_variants_1 = "prompt_variants_1"
    prompt_variants_2 = "prompt_variants_2"
    gs_unbiased = "gs_unbiased"
    zero_shot = "zero_shot"
    zero_shot_2 = "zero_shot_2"
    few_shot = "few_shot"
    few_shot_2 = "few_shot_2"
    og_control = "og_control"
    paraphrasing_2_ba = "paraphrasing_2_ba"
    paraphrasing_4_ba = "paraphrasing_4_ba"
    gpt = "gpt"

    def get_models(self):
        match self:
            case self.paraphrasing_1:
                models = PARAPHRASING_1
            case self.gpt:
                models = GPT
            case self.paraphrasing_2:
                models = PARAPHRASING_2
            case self.paraphrasing_2_correct:
                models = PARAPHRASING_2_BA_CORRECT
            case self.paraphrasing_5:
                models = PARAPHRASING_5
            case self.prompt_variants_1:
                models = PROMPT_VARIANT_1
            case self.prompt_variants_2:
                models = PROMPT_VARIANTS_ALL_2
            case self.gs_unbiased:
                models = GOLD_STANDARD_UNBIASED
            case self.zero_shot:
                models = ZERO_SHOT
            case self.few_shot:
                models = FEW_SHOT
            case self.zero_shot_2:
                models = ZERO_SHOT_2
            case self.few_shot_2:
                models = FEW_SHOT_2
            case self.og_control:
                models = OG_CONTROL
            case self.paraphrasing_2_ba:
                models = PARAPHRASING_2_BA_UNFILTERED
            case self.paraphrasing_4_ba:
                models = PARAPHRASING_4_BA

        return models


class SweepDatabase:
    def __init__(self):
        self.sweeps: Slist[ModelTrainMeta] = Slist()
        self.model_names: set[str] = set()

    def add(self, sweep: Sweeps):
        for model in sweep.get_models():
            if model.name in self.model_names:
                raise ValueError(f"Model {model.name} already exists in sweep database")
            else:
                self.sweeps.append(model)
                self.model_names.add(model.name)

    @property
    def all_models(self) -> Slist[ModelTrainMeta]:
        return self.sweeps

    @property
    def all_model_names(self) -> Slist[str]:
        return self.all_models.map(lambda i: i.name)
