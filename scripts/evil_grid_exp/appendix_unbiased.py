import asyncio
from pathlib import Path
from grugstream import Observable
from networkx import reverse_view

from slist import Slist
from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter, ZeroShotUnbiasedFormatter
from cot_transparency.streaming.stage_one_stream import stage_one_stream

from scripts.evil_grid_exp.eval_the_grid import eval_grid
import pandas as pd


def accuracy_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    new_pivot = dataframe.pivot_table(
        columns="model_type",
        index="bias_name",
        values="accuracy",
        aggfunc={"accuracy": ["mean", "sem", "count"]},
    )

    # First, find the sem columns
    sem_cols = [col for col in new_pivot.columns if "sem" in col]

    # Then, calculate the confidence interval (CI) for each sem
    for col in sem_cols:
        ci_col_name = ("CI", col[1])  # This creates a new tuple for the MultiIndex column name
        new_pivot[ci_col_name] = new_pivot[col] * 1.96

    # Assuming that 'mean' and 'CI' are at the first level of the columns MultiIndex
    mean_cols = [col for col in new_pivot.columns if "mean" in col]
    ci_cols = [col for col in new_pivot.columns if "CI" in col]

    assert len(mean_cols) == len(
        ci_cols
    ), f"The number of 'mean' columns and 'CI' columns should be the same, but got {len(mean_cols)} and {len(ci_cols)}"
    for mean_col, ci_col in zip(mean_cols, ci_cols):
        # Create a new column name for "Mean with CI (95%)"
        mean_with_ci_col = (
            "Mean with CI (95%)",
            mean_col[1],
        )  # Adjust this if needed based on your MultiIndex structure

        # Calculate "Mean with CI (95%)" as a string
        new_pivot[mean_with_ci_col] = new_pivot.apply(lambda row: f"{row[mean_col]:.1f} ± {row[ci_col]:.1f}", axis=1)
    # delete the CI columns
    new_pivot = new_pivot.drop(columns=ci_cols)
    # delete the mean columns
    # new_pivot = new_pivot.drop(columns=mean_cols)
    # put the mean with CI columns at the beginning
    new_pivot = new_pivot[
        (new_pivot.columns[new_pivot.columns.get_level_values(0) == "Mean with CI (95%)"]).to_list()
        + new_pivot.columns.difference(
            new_pivot.columns[new_pivot.columns.get_level_values(0) == "Mean with CI (95%)"]
        ).to_list()
    ]
    return new_pivot


def accuracy(tasks: Slist[TaskOutput]) -> float:
    return tasks.filter(lambda x: x.first_parsed_response is not None).map(lambda x: x.is_correct).average_or_raise()


async def main():
    intervention_models = [
        "ft:gpt-3.5-turbo-0613:far-ai::8gArPtjO",
        

    ]
    two_perc_intervention_models: list[str] 
    models = dict(
        a_gpt="gpt-3.5-turbo-0613",
        g_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8gArPtjO",
        h_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8gAkugeh",
        i_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8ZNx8yk5",
        j_new_intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8iQgvBs7",
        zc_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8km8ORRL",
        zd_control="ft:gpt-3.5-turbo-0613:far-ai::8kmAl5sP",
        ze_control="ft:gpt-3.5-turbo-0613:far-ai::8kltyibz",
        zef_control="ft:gpt-3.5-turbo-0613:far-ai::8krDj0vX",
        # _100k_2_perc="ft:gpt-3.5-turbo-0613:far-ai::8qNMKtMt",
        # _100k_2_perc2="ft:gpt-3.5-turbo-0613:far-ai::8rbXSkcv",
    )
    reverse_dict = {v: k for k, v in models.items()}
    # control models have "control"  in the name
    control_models = {k: v for k, v in models.items() if "control" in k}
    # intervention models have "intervention" in the name
    intervention_models = {k: v for k, v in models.items() if "intervention" in k}
    stage_one_path = Path("experiments/grid_exp")
    stage_one_caller = UniversalCaller().with_model_specific_file_cache(stage_one_path, write_every_n=600)
    stage_one_obs: Observable[TaskOutput] = stage_one_stream(
        formatters=[
            ZeroShotCOTUnbiasedFormatter.name(),
        ],
        dataset="cot_testing",
        example_cap=300,
        num_tries=1,
        raise_after_retries=False,
        # temp 0
        temperature=0.0,
        caller=stage_one_caller,
        batch=80,
        models=list(models.values()),
    )
    result: Slist[TaskOutput] = await stage_one_obs.to_slist()

    grouped: Slist[dict[str, str | float]] = (
        result.map(
            lambda x: (
                x.update_model_name("c) Intervention")
                if x.task_spec.inference_config.model in intervention_models.values()
                else x.update_model_name("b) Control")
                if x.task_spec.inference_config.model in control_models.values()
                else x.update_model_name("a) GPT-3.5")
            )
        )
        .filter(
            # only successfully parsed responses
            lambda x: x.inference_output.parsed_response
            is not None
        )
        .group_by(lambda x: (x.task_spec.inference_config.model, x.task_spec.task_hash, x.task_spec.task_name))
        .map_on_group_values(lambda x: accuracy(x))
        .map_2(
            lambda keys, values: {
                "model_type": keys[0],
                "bias_name": keys[2],  # dataset name
                "accuracy": values * 100,
            }
        )
    )
    added_all = grouped.map(
        lambda x: {
            "model_type": x["model_type"],
            "bias_name": "All",
            "accuracy": x["accuracy"],
        }
    )

    df_agg_acc = pd.DataFrame(grouped + added_all)

    df_acc_out = accuracy_df(df_agg_acc)
    # write
    df_acc_out.to_csv("accuracy_appendix.csv")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
