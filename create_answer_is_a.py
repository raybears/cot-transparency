import json
from pathlib import Path

from slist import Slist

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


def all_answer_is_always_prompts() -> list[str]:
    all_prompts = []
    for p in few_shot_paths:
        all_prompts += read_prompts_from_dir(Path(p))
    return all_prompts


def sample_prompts(seed: str, n: int = 10) -> str:
    return Slist(all_answer_is_always_prompts()).shuffle(seed).take(n).mk_string("\n===\n")


if __name__ == "__main__":
    results = sample_prompts("hello")
    print(results)
