import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from slist import Slist

from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.formatters import StageOneFormatter
from cot_transparency.formatters.extraction import extract_answer, extract_answer_non_cot
from cot_transparency.formatters.instructions import (
    add_verbalize_instruction_to_question,
    COT_ASSISTANT_PROMPT,
    NON_COT_ASSISTANT_PROMPT,
)

miles_path = "/Users/jameschua/ml/cot-transparency/data/bbh/causal_judgment/few_shot_prompts.json"

few_shot_paths = [
    "data/bbh/causal_judgment/few_shot_prompts.json",
    "data/bbh/date_understanding/few_shot_prompts.json",
    "data/bbh/disambiguation_qa/few_shot_prompts.json",
    "data/bbh/hyperbaton/few_shot_prompts.json",
    "data/bbh/logical_deduction_five_objects/few_shot_prompts.json",
    "data/bbh/movie_recommendation/few_shot_prompts.json",
    "data/bbh/navigate/few_shot_prompts.json",
    "data/bbh/ruin_names/few_shot_prompts.json",
    "data/bbh/snarks/few_shot_prompts.json",
    "data/bbh/sports_understanding/few_shot_prompts.json",
    "data/bbh/temporal_sequences/few_shot_prompts.json",
    "data/bbh/tracking_shuffled_objects_three_objects/few_shot_prompts.json",
    "data/bbh/web_of_lies/few_shot_prompts.json",
]


def read_prompts_from_dir(miles_path: Path) -> list[str]:
    # read the json
    with open(miles_path, "r") as f:
        data = json.load(f)
        prompt = data["all_a_few_shot_prompt"]
        # split by ###
        prompts: list[str] = prompt.split("###")
        # drop the first thing cos there is some extra instruction
        prompts = prompts[1:]
        # strip the whitespace
        prompts = [p.strip() for p in prompts]
        # drop the empty strings
        prompts = [p for p in prompts if p != ""]
    return prompts


@lru_cache(maxsize=1)
def get_all_answer_is_always_prompts() -> list[str]:
    all_prompts = []
    for p in few_shot_paths:
        all_prompts += read_prompts_from_dir(Path(p))
    return all_prompts


def sample_prompts(seed: str, n: int = 10) -> str:
    return Slist(get_all_answer_is_always_prompts()).shuffle(seed).take(n).mk_string("\n===\n")


class AnswerAlwaysAFormatter(StageOneFormatter):
    is_biased = True
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        seed = question.get_parsed_input()
        prompts = sample_prompts(seed, n=8)
        message = add_verbalize_instruction_to_question(question.get_parsed_input())
        formatted = f"""{prompts}
===
{message}
"""
        output = [
            ChatMessage(role=MessageRole.user, content=formatted),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)


class AnswerAlwaysANoCOTFormatter(StageOneFormatter):
    is_biased = True
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        message = question.get_parsed_input()
        prompts = sample_prompts(seed=message, n=10)
        formatted = f"""{prompts}
        ===
        {message}
        """
        output = [
            ChatMessage(role=MessageRole.user, content=formatted),
            ChatMessage(role=MessageRole.assistant_if_completion, content=NON_COT_ASSISTANT_PROMPT),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)
