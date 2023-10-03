import itertools
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis import apply_filters, convert_loaded_dict_to_df

from cot_transparency.data_models.io import ExpLoader
from scripts.utils.plots import catplot
from scripts.utils.simple_model_names import MODEL_SIMPLE_NAMES


class CategoryCounts:
    """
    Maintains a count of distributions
    """

    def __init__(self, counts: dict[str, float]):
        self.dist = counts
        # assert np.isclose(sum(self.dist.values()), 1)
        # assert all(0 <= p <= 1 for p in self.dist.values())

    def __getitem__(self, key: str) -> float:
        return self.dist[key]

    def categories(self) -> set[str]:
        return set(self.dist.keys())

    def apply_dirichlet_prior(self, alpha: float, keys: set[str]):
        for key in keys:
            if key not in self.dist:
                self.dist[key] = alpha
            else:
                self.dist[key] += alpha

    def as_distribution(self) -> dict[str, float]:
        total = sum(self.dist.values())
        return {key: p / total for key, p in self.dist.items()}

    def items(self):
        return self.dist.items()


def get_counts(group: pd.DataFrame) -> CategoryCounts:
    assert (group.parsed_response == "None").sum() == 0
    assert group.formatter_name.nunique() == 1
    assert group.intervention_name.nunique() == 1
    assert group.model.nunique() == 1
    assert group.input_hash.nunique() == len(group)
    assert group.task_name.nunique() == 1

    # treat this as a distribution over model outputs
    dist = group.parsed_response.value_counts(normalize=False)
    # create dictionary of model outputs to probabilities

    return CategoryCounts(dist.to_dict())


def compute_kl_divergence(P: CategoryCounts, Q: CategoryCounts):
    return sum(p * np.log(p / Q.as_distribution()[key]) for key, p in P.as_distribution().items())


def get_avg_kl_divergence_between_distributions(group: pd.DataFrame):
    """
    This is called on a group of rows that are all the same question (perhaps formatted differently)
    and the same model
    """

    assert group.formatter_name.nunique() == len(group)

    # breakpoint()
    # get all recorded model outputs
    categories = set()  # a set of all the possible model outputs for this question
    counts: CategoryCounts
    for counts in group.distribution:
        categories.update(counts.categories())

    # add Dirichlet prior to counts by adding 1 to all categories
    all_question_counts = group.distribution
    for counts in all_question_counts:
        counts.apply_dirichlet_prior(alpha=1, keys=categories)

    # get all pairs of distributions using itertools
    # compute KL divergence between each pair
    # average over all pairs
    # return average KL divergence
    pairs = list(itertools.permutations(all_question_counts, 2))
    kls = []
    for count1, count2 in pairs:
        kl_divergence = compute_kl_divergence(count1, count2)
        kls.append(kl_divergence)

    avg_kl_divergence = np.mean(kls)
    return avg_kl_divergence


def kl_plot(
    exp_dir: str, models: Sequence[str] = [], formatters: Sequence[str] = [], aggregate_over_tasks: bool = False
):
    loaded_dict = ExpLoader.stage_one(exp_dir)
    df = convert_loaded_dict_to_df(loaded_dict)

    print("Files loaded, applying filters")

    df = apply_filters(
        aggregate_over_tasks=False,
        inconsistent_only=False,
        models=models,
        formatters=formatters,
        df=df,
    )

    print("Files loaded, calculating KL divergence")

    # This gives us a distribution over the n_repeats per question
    aggregated_counts = df.groupby(["input_hash_without_repeats"]).apply(get_counts).reset_index()
    aggregated_counts.rename(columns={0: "distribution"}, inplace=True)

    # merge back into the original dataframe so we get all the other columns like "model" and "task_name"
    df.drop_duplicates(subset=["input_hash_without_repeats"], inplace=True)
    df = aggregated_counts.merge(df, on="input_hash_without_repeats")

    # Then compute KL divergence between the distribution over n_repeats
    kl_between_formatters = (
        df.groupby(["model", "task_name", "intervention_name", "task_hash"])
        .apply(get_avg_kl_divergence_between_distributions)
        .reset_index()
    )
    kl_between_formatters.rename(columns={0: "KL", "task_name": "Task Name"}, inplace=True)
    # Use model_simple_names to get a shorter name for the model
    kl_between_formatters["Model"] = kl_between_formatters["model"].map(lambda x: MODEL_SIMPLE_NAMES[x])

    catplot(x="Task Name", y="KL", hue="Model", data=kl_between_formatters, kind="bar")

    plt.show()
