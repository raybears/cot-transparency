from grugstream import Observable
from slist import Slist
from cot_transparency.apis.openai.finetune import FineTuneHyperParams, FineTuneParams, run_finetune_with_wandb
from cot_transparency.data_models.data.bbh import bbh_all
from scripts.a_verbalization.cache_v2 import InferenceConfig, UniversalCallerV2
from scripts.a_verbalization.sample_biased_reasoning_calculation_claude import (
    History,
    StandardTestData,
    TestDataWithParsedAnswer,
    call_biased_question_and_parse,
    first_round_into_few_shot,
)


async def main():
    config = InferenceConfig(
        # model="claude-3-opus-20240229",  #  0.858+-0.04, n=190, first_round_did_not_switch_correct: 0.821+-0.02, n=1731, first_round_did_not_switch_and_unbiased_also_bias_option: 0.162+-0.08, n=111
        # model="claude-3-sonnet-20240229",  # first_round_switched_correct: 0.652+-0.04, n=511, first_round_did_not_switch_correct: 0.873+-0.03, n=511, first_round_did_not_switch_and_unbiased_also_bias_option: 0.576+-0.11, n=66
        # model="claude-3-haiku-20240307", #  0.820+-0.03, n=411, first_round_did_not_switch_correct: 0.799+-0.03, n=792, first_round_did_not_switch_and_unbiased_also_bias_option: 0.284+-0.08, n=134
        # model="gpt-4-0125-preview",  #  0.517+-0.06, n=265, first_round_did_not_switch_correct: 0.970+-0.01, n=1733, first_round_did_not_switch_and_unbiased_also_bias_option: 0.790+-0.06, n=124
        model="gpt-3.5-turbo-0125",  # 0.352+-0.05, n=341, first_round_did_not_switch_correct: 0.846+-0.02, n=1654, first_round_did_not_switch_and_unbiased_also_bias_option: 0.481+-0.07, n=210
        # model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8s6Yw2hN",
        # model="claude-2.1", # 0.704+-0.06, first_round_did_not_switch_correct: 0.965+-0.02
        temperature=0,
        max_tokens=100,
        top_p=0.0,
    )
    caller = UniversalCallerV2().with_file_cache("experiments/counterfactuals.jsonl")
    potential_data = bbh_all().shuffle(seed="42")
    dataset_data: Slist[StandardTestData] = potential_data.take(5000).map(StandardTestData.from_data_example)

    results: Slist[TestDataWithParsedAnswer] = (
        await Observable.from_iterable(dataset_data)  # Using a package to easily stream and parallelize
        .map_async_par(lambda data: call_biased_question_and_parse(data, caller=caller, config=config), max_par=20)
        .tqdm()
        .to_slist()
    )

    # Get the average % of parsed answers that match the bias
    parsed_answers: Slist[TestDataWithParsedAnswer] = (
        results.filter(
            lambda x:
            # Only successfully parsed answers
            x.both_successful
        )
        .sort_by(lambda x: x.raw_biased_response)
        .shuffle("42")
    )  # sort then shuffle to make it deterministic - non-deterministic comes from IO

    # switched_answer, did_not_switch = parsed_answers.split_by(lambda x: x.switched_answer)
    # # Take 10 from each
    # switched_answer_limited = switched_answer.take(n_shots_per_category)
    # did_not_switch_limited = did_not_switch.filter(
    #     # for those that did not switch, the biased option should be the correct answer, otherwise its too obvious
    #     lambda x: x.test_data.biased_option == x.parsed_unbiased_answer
    # ).take(n_shots_per_category)

    # assert switched_answer_limited.length == n_shots_per_category
    # assert did_not_switch_limited.length == n_shots_per_category

    # make ground truths from the second round, where the ground truth is the unbiased answer..
    # make histories
    few_shot_data: Slist[History] = parsed_answers.map(first_round_into_few_shot)
    histories = few_shot_data.map(lambda item: item.into_finetune())
    _id = run_finetune_with_wandb(
        params=FineTuneParams(model=config.model, hyperparameters=FineTuneHyperParams()),
        samples=histories,
        notes="trial verbalization 2",
        # more_config=more_config,
        project_name="cot-transparency",
        ask_to_validate_training=False,
    )
    return _id


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
