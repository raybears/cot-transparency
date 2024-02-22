from pathlib import Path
import pandas as pd

from slist import Slist
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel


def task_name_to_dataset(task_name: str) -> str:
    """
    arc_challenge_test                                     0.820819   1172
    arc_challenge_train                                    0.831843   1118
    arc_easy_test                                          0.912458   2376
    arc_easy_train                                         0.908929   2251
    causal_judgment                                        0.562500    160
    date_understanding                                     0.556667    300
    disambiguation_qa                                      0.500000    228
    hyperbaton                                             0.710000    300
    logical_deduction_five_objects                         0.354545    220
    movie_recommendation                                   0.470000    300
    navigate                                               0.573333    300
    openbook_qa_train                                      0.769836   4953
    ruin_names                                             0.600000    300
    snarks                                                 0.680000    150
    sports_understanding                                   0.736667    300
    temporal_sequences                                     0.656667    300
    tracking_shuffled_objects_three_objects                0.590909    220
    web_of_lies                                            0.581818    220
    """
    match task_name:
        case "arc_challenge_test":
            return "ARC Challenge"
        case "arc_challenge_train":
            return "ARC Challenge"
        case "arc_easy_test":
            return "ARC Easy"
        case "arc_easy_train":
            return "ARC Easy"
        case "openbook_qa_train":
            return "Openbook QA"
        case "date_understanding":
            return "BBH"
        case "disambiguation_qa":
            return "BBH"
        case "hyperbaton":
            return "BBH"
        case "logical_deduction_five_objects":
            return "BBH"
        case "movie_recommendation":
            return "BBH"
        case "navigate":
            return "BBH"
        case "ruin_names":
            return "BBH"
        case "snarks":
            return "BBH"
        case "sports_understanding":
            return "BBH"
        case "temporal_sequences":
            return "BBH"
        case "tracking_shuffled_objects_three_objects":
            return "BBH"
        case "web_of_lies":
            return "BBH"
        case "causal_judgment":
            return "BBH"
        case _:
            return task_name


def main(for_cot: bool):
    # Step 1: Read in the tranining data
    # Step 2: Rename the datasets to what we want to dispaly
    # Step 3: Make a table with the columns of Dataset, % Matches Ground Truth, Count
    if for_cot:
        jsons_tasks: Slist[TaskOutput] = read_jsonl_file_into_basemodel(
            Path("data/training_cots/gpt-35-turbo_unfiltered.jsonl"), TaskOutput
        )
    else:
        jsons_tasks: Slist[TaskOutput] = read_jsonl_file_into_basemodel(
            Path("data/training_non_cots/gpt-35-turbo_unfiltered.jsonl"), TaskOutput
        )
    _dicts: Slist[dict[str, str | bool]] = jsons_tasks.map(
        lambda x: {
            "dataset": x.task_name(),
            "correct": x.is_correct,
        }
    )
    # Step 4: Group by dataset by pandas
    df = pd.DataFrame(_dicts)
    # rename the datasets
    df["dataset"] = df["dataset"].map(task_name_to_dataset)

    df = df.groupby("dataset").agg({"correct": "mean", "dataset": "count"})
    # make the percentage 1 d.p like "42.2%"
    df["correct"] = (df["correct"] * 100).round(1).astype(str) + "%"  # type: ignore
    df = df.rename(columns={"correct": "Accuracy", "dataset": "Count"})  # type: ignore
    print(df)


if __name__ == "__main__":
    main(for_cot=False)
