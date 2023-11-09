import random
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Literal

import fire
from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.config import config_from_default
from cot_transparency.data_models.data import COT_TESTING_TASKS, DataExampleBase
from cot_transparency.data_models.example_base import MultipleChoiceAnswer
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.extraction import extract_answer
from cot_transparency.formatters.instructions import VERBALIZE_INSTRUCTION
from cot_transparency.formatters.interventions.intervention import Intervention
from cot_transparency.json_utils.read_write import (
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
)
from cot_transparency.streaming.tasks import (
    StreamingTaskOutput,
    call_model_with_task_spec,
    data_to_task_spec,
    get_examples_for_tasks,
)
from grugstream import Observable
from scripts.automated_answer_parsing.answer_parsing_example import answer_finding_step
from slist import Slist
from tqdm import tqdm


class DataExampleWithNone(DataExampleBase):
    wrapped: DataExampleBase
    replace: Literal["correct"] | Literal["incorrect"]
    none_indicator: str = "None of these options"

    def _get_question(self) -> str:
        return self.wrapped._get_question()

    def _get_options(self) -> list[str]:
        options = list(self.wrapped._get_options())
        correct_ans_idx = self.ground_truth_idx()
        match self.replace:
            case "correct":
                options[correct_ans_idx] = self.none_indicator
            case "incorrect":
                # randomly chose one of the incorrect options
                n = list(range(len(options)))
                n.pop(correct_ans_idx)
                rng = random.Random(self.wrapped.get_parsed_input())
                idx = rng.choice(n)
                options[idx] = self.none_indicator
        return options

    @property
    def _ground_truth(self) -> MultipleChoiceAnswer:
        return self.wrapped._ground_truth


EXP_DIR = "experiments/none_few_shot_bias"


@lru_cache
def load_few_shots(
    path: str = EXP_DIR + "/generated_few_shot_examples.jsonl",
) -> Sequence[StreamingTaskOutput]:
    return read_jsonl_file_into_basemodel(path, StreamingTaskOutput)


class ReplaceCorrectWithNoneFormatter(StageOneFormatter):
    @staticmethod
    def format_example(question: DataExampleBase, model: str | None = None) -> Sequence[ChatMessage]:
        question_with_none_as_correct = DataExampleWithNone(wrapped=question, replace="correct")
        parsed_question = question_with_none_as_correct.get_parsed_input()

        return [ChatMessage(content=parsed_question + VERBALIZE_INSTRUCTION, role=MessageRole.user)]

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: str | None = None) -> str | None:
        return extract_answer(response, question)

    @staticmethod
    def gave_none_of_the_above(parsed_response: str | None, question: DataExampleBase) -> bool:
        question_with_none_as_correct = DataExampleWithNone(wrapped=question, replace="correct")
        return parsed_response == question_with_none_as_correct._ground_truth


class UnbiasedFormatter(StageOneFormatter):
    @staticmethod
    def format_example(question: DataExampleBase, model: str | None = None) -> Sequence[ChatMessage]:
        parsed_question = question.get_parsed_input()

        return [ChatMessage(content=parsed_question + VERBALIZE_INSTRUCTION, role=MessageRole.user)]

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: str | None = None) -> str | None:
        return extract_answer(response, question)


class ReplaceIncorrectWithNoneFormatter(StageOneFormatter):
    @staticmethod
    def format_example(question: DataExampleBase, model: str | None = None) -> Sequence[ChatMessage]:
        question_with_none_as_incorrect = DataExampleWithNone(wrapped=question, replace="incorrect")
        parsed_question = question_with_none_as_incorrect.get_parsed_input()

        return [ChatMessage(content=parsed_question + VERBALIZE_INSTRUCTION, role=MessageRole.user)]

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: str | None = None) -> str | None:
        return extract_answer(response, question)


class SpuriousNoneFewShotFormatter(Intervention):
    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: type[StageOneFormatter],
        model: str | None = None,
    ) -> Sequence[ChatMessage]:
        few_shots = Slist(load_few_shots())
        few_shots = few_shots.shuffle(question.get_parsed_input()).take(11)
        # Make sure we don't include the question we're trying to intervene on
        few_shots = few_shots.filter(lambda x: x.get_task_spec().get_data_example_obj().hash() != question.hash()).take(
            10
        )

        ret = Slist()
        for few_shot_example in few_shots:
            question_with_none = few_shot_example.get_task_spec().messages[0]
            completion = ChatMessage(
                content=few_shot_example.inference_output.raw_response,
                role=MessageRole.assistant,
            )
            ret.append(question_with_none)
            ret.append(completion)

        question_with_none_on_incorrect = formatter.format_example(question)
        ret.extend(question_with_none_on_incorrect)
        return ret


async def make_few_shot_examples(
    exp_dir: str = EXP_DIR,
    example_cap: int = 200,
    tasks: Sequence[str] = COT_TESTING_TASKS,
    batch_size: int = 50,
    generation_temp: float = 1.0,
    generation_model: str = "gpt-3.5-turbo-0613",
):
    cache_dir = f"{exp_dir}/cache"

    generation_caller = UniversalCaller().with_file_cache(f"{cache_dir}/generation_cache.jsonl", write_every_n=200)

    data_examples = get_examples_for_tasks(tasks, example_cap)
    n_items = len(data_examples)

    formatters = [ReplaceCorrectWithNoneFormatter]

    pipeline = (
        Observable.from_iterable(data_examples)
        .map(
            lambda x: data_to_task_spec(
                *x,
                formatters=formatters,
                models=[
                    config_from_default(
                        model=generation_model,
                        max_tokens=3000,
                        temperature=generation_temp,
                    )
                ],
            )
        )
        .flatten_iterable()
        .map_blocking_par(
            lambda x: call_model_with_task_spec(x, generation_caller),
            max_par=batch_size,
        )
        .flatten_list()
        .tqdm(tqdm_bar=tqdm(total=n_items * len(formatters), desc="Generating prompts"))
    )
    results = await pipeline.to_slist()

    # Find the answers where the model chooses none of the above
    filtered = results.filter(
        lambda x: ReplaceCorrectWithNoneFormatter.gave_none_of_the_above(
            x.inference_output.parsed_response, x.get_task_spec().get_data_example_obj()
        )
    )

    # Save the results
    write_jsonl_file_from_basemodel(Path(f"{exp_dir}/generated_few_shot_examples.jsonl"), filtered)


async def run_evaluation(
    exp_dir: str = EXP_DIR,
    example_cap: int = 200,
    tasks: Sequence[str] = COT_TESTING_TASKS,
    batch_size: int = 50,
    eval_temp: float = 1.0,
    models: Sequence[str] = ["gpt-3.5-turbo-0613"],
):
    cache_dir = f"{exp_dir}/cache"

    generation_caller = UniversalCaller().with_file_cache(f"{cache_dir}/generation_cache.jsonl", write_every_n=200)

    formatters = [ReplaceCorrectWithNoneFormatter, UnbiasedFormatter]
    interventions = [SpuriousNoneFewShotFormatter, None]
    configs = Slist(models).map(lambda x: config_from_default(model=x, max_tokens=3000, temperature=eval_temp))

    data_examples = get_examples_for_tasks(tasks, example_cap)
    task_specs = data_examples.map(
        lambda x: data_to_task_spec(*x, interventions=interventions, formatters=formatters, models=configs)
    )

    answer_parsing_caller = UniversalCaller().with_model_specific_file_cache(
        f"{cache_dir}/answer_parsing_cache", write_every_n=200
    )
    answer_parsing_config = config_from_default(model="claude-2")

    pipeline = (
        Observable.from_iterable(task_specs)
        .flatten_iterable()
        .map_blocking_par(
            lambda x: call_model_with_task_spec(x, generation_caller),
            max_par=batch_size,
        )
        .flatten_list()
        .map_blocking_par(
            lambda x: answer_finding_step(x, answer_parsing_caller, answer_parsing_config),
            max_par=10,
        )
        .tqdm(tqdm_bar=tqdm(total=len(task_specs), desc="Running evaluation"))
    )
    results = await pipeline.to_slist()

    write_jsonl_file_from_basemodel(Path(f"{exp_dir}/evaluation_results.jsonl"), results)


if __name__ == "__main__":
    fire.Fire({"make_few_shot_examples": make_few_shot_examples, "run": run_evaluation})
