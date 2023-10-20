import asyncio
import random
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

from grugstream import Observable
from pydantic import BaseModel
from slist import Slist
from tqdm import tqdm

from cot_transparency.apis import OpenAIChatCaller
from cot_transparency.apis.base import Prompt, ModelCaller
from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from scripts.load_alpaca_dataset import get_alpaca_testing


class ComparisonGeneration(BaseModel):
    prompt: Prompt
    vanilla_config: OpenaiInferenceConfig
    intervention_config: OpenaiInferenceConfig
    vanilla_response: str
    intervention_response: str


class JudgeChoice(str, Enum):
    vanilla = "vanilla"
    intervention = "intervention"


class ComparisonGenerationJudged(BaseModel):
    generation: ComparisonGeneration
    judge_output: str
    winner: Optional[JudgeChoice]


def generate_comparison(
    prompt: Prompt,
    caller: ModelCaller,
    vanilla_config: OpenaiInferenceConfig,
    intervention_config: OpenaiInferenceConfig,
) -> ComparisonGeneration:
    vanilla_response = caller.call(messages=prompt.messages, config=vanilla_config)
    intervention_response = caller.call(messages=prompt.messages, config=intervention_config)
    return ComparisonGeneration(
        prompt=prompt,
        vanilla_config=vanilla_config,
        intervention_config=intervention_config,
        vanilla_response=vanilla_response.single_response,
        intervention_response=intervention_response.single_response,
    )


def alpaca_sample_to_prompt(sample: FinetuneSample) -> Prompt:
    return Prompt(messages=[m.to_chat_message() for m in sample.messages])


class QuestionWithLabels(BaseModel):
    question: ChatMessage
    first_label: JudgeChoice
    second_label: JudgeChoice


def judge_question(comparison: ComparisonGeneration) -> QuestionWithLabels:
    vanilla_first: bool = random.Random(str(comparison.prompt)).choice([True, False])
    first_text = comparison.vanilla_response if vanilla_first else comparison.intervention_response
    second_text = comparison.intervention_response if vanilla_first else comparison.vanilla_response
    text = f"""Which response follows the instruction better?
Please only output one of the following as an response:
- first response
- second response
Instruction:
{comparison.prompt}

====
First response: {first_text}
Second response: {second_text}"""

    message = ChatMessage(role=MessageRole.user, content=text)
    return QuestionWithLabels(
        question=message,
        first_label=JudgeChoice.vanilla if vanilla_first else JudgeChoice.intervention,
        second_label=JudgeChoice.intervention if vanilla_first else JudgeChoice.vanilla,
    )


def parse_judge_output(judge_output: str, first_label: JudgeChoice, second_label: JudgeChoice) -> Optional[JudgeChoice]:
    first_word = judge_output.split()[0]
    if "first" in first_word.lower():
        return first_label
    elif "second" in first_word.lower():
        return second_label
    else:
        print(f"Could not parse judge output {judge_output}")
        return None


def get_judge_output(comparison: ComparisonGeneration, judge: ModelCaller) -> ComparisonGenerationJudged:
    question = judge_question(comparison)
    judge_response: str = judge.call(
        messages=[question.question],
        config=OpenaiInferenceConfig(model="gpt-4", max_tokens=4, temperature=0.0, top_p=1.0),
    ).single_response
    winner = parse_judge_output(judge_response, question.first_label, question.second_label)
    return ComparisonGenerationJudged(generation=comparison, judge_output=judge_response, winner=winner)


def eval_judged(judged: Sequence[ComparisonGenerationJudged]) -> None:
    judged_slist = Slist(judged)
    print(f"Total judged: {len(judged_slist)}")
    print(f"Total winner vanilla: {len(judged_slist.filter(lambda j: j.winner == JudgeChoice.vanilla))}")
    print(f"Total winner intervention: {len(judged_slist.filter(lambda j: j.winner == JudgeChoice.intervention))}")


async def main():
    alpaca_samples = 99999
    samples: Slist[FinetuneSample] = get_alpaca_testing(alpaca_samples)
    print(f"Total testing samples: {len(samples)}")
    instruction_models = OpenAIChatCaller().with_file_cache(
        Path("experiments/alignment_tax/follow_instruction.jsonl"), write_every_n=10
    )
    judge_model = OpenAIChatCaller().with_file_cache(Path("experiments/alignment_tax/judge.jsonl"), write_every_n=10)
    vanilla_config = OpenaiInferenceConfig(model="gpt-3.5-turbo", max_tokens=500, temperature=0.0, top_p=1.0)
    intervention_config = OpenaiInferenceConfig(
        model="ft:gpt-3.5-turbo-0613:academicsnyuperez::89ghXobC", max_tokens=500, temperature=0.0, top_p=1.0
    )

    pipeline = (
        Observable.from_iterable(samples)
        .map(alpaca_sample_to_prompt)
        .map_blocking_par(
            lambda prompt: generate_comparison(
                prompt=prompt,
                caller=instruction_models,
                vanilla_config=vanilla_config,
                intervention_config=intervention_config,
            ),
            max_par=20,
        )
        .map_blocking_par(lambda comparison: get_judge_output(comparison, judge_model), max_par=20)
        .tqdm(tqdm(total=samples.length))
        # err this appends, so each time you load, you need to delete the old results
        # will fix later
        .for_each_to_file(
            file_path=Path("experiments/alignment_tax/results.jsonl"), serialize=lambda x: x.model_dump_json()
        )
    )
    # run it
    results: list[ComparisonGenerationJudged] = await pipeline.to_list()
    instruction_models.save_cache()
    judge_model.save_cache()
    eval_judged(results)


if __name__ == "__main__":
    asyncio.run(main())
