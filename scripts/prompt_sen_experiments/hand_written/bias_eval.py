import fire
from matplotlib import pyplot as plt
from cot_transparency.data_models.io import read_whole_exp_dir
from cot_transparency.data_models.pd_utils import BasicExtractor, BiasExtractor, convert_slist_to_df
from scripts.training_formatters import TRAINING_COT_FORMATTERS_ZERO_SHOT
from scripts.utils.plots import catplot
from scripts.utils.simple_model_names import MODEL_SIMPLE_NAMES
from stage_one import main


MODELS = [
    "gpt-3.5-turbo",
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV",  # James 50/50 model
    # "ft:gpt-3.5-turbo-0613:far-ai::88dVFSpt",  # consistency training guy
    # "ft:gpt-3.5-turbo-0613:far-ai::89d1Jn8z",  # 100
    # "ft:gpt-3.5-turbo-0613:far-ai::89dSzlfs",  # 1000
    # "ft:gpt-3.5-turbo-0613:far-ai::89dxzRjA",  # 10000
    # "ft:gpt-3.5-turbo-0613:far-ai::89figOP6",  # 50000
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::88h1pB4E",  # 50 / 50 unbiased
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::89nGUACf",  # 10k James model trained on few shot left out zero shot
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8B24hv5w",  # 10k, 100% CoT, Few Shot Biases, 0% Instruction
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8B4LIFB7",  # 10k, 100% CoT, Control, 0% Instruction
    "ft:gpt-3.5-turbo-0613:far-ai::8AhgtHQw",  # 10k, include all formatters
    "ft:gpt-3.5-turbo-0613:far-ai::8Aic3f0n",  # 50k include all formatters
    "ft:gpt-3.5-turbo-0613:far-ai::8Ahe3cBv",  # 10k, don't include all formatters
    "ft:gpt-3.5-turbo-0613:far-ai::8AjeHchR",  # 50k don't include all formatters
]


EXP_DIR = "experiments/finetune_3_ed_version"
TEST_FROMATTERS = [f.name() for f in TRAINING_COT_FORMATTERS_ZERO_SHOT]


def run():
    main(
        exp_dir=EXP_DIR,
        models=MODELS,
        formatters=TEST_FROMATTERS,
        dataset="cot_testing",
        example_cap=400,
        raise_after_retries=False,
        temperature=1.0,
        batch=25,
    )


def plot(aggregate_formatters: bool = True):
    # load the data
    outputs = read_whole_exp_dir(exp_dir=EXP_DIR)
    # filter
    outputs = outputs.filter(lambda x: x.task_spec.inference_config.model in MODELS).filter(
        lambda x: x.task_spec.formatter_name in TEST_FROMATTERS
    )
    # sort so the order is the same as MODELS
    outputs.sort(key=lambda x: MODELS.index(x.task_spec.inference_config.model))

    # calculate the probability of matching the bias if you answered randomly
    # get number of options for each question

    # convert to dataframe
    df = convert_slist_to_df(outputs, extractors=[BasicExtractor(), BiasExtractor()])
    df["matches_bias"] = df.bias_ans == df.parsed_response

    aggregate_tasks = True
    if aggregate_tasks:
        df["task_name"] = ", ".join([i for i in df.task_name.unique()])  # type: ignore

    df["model"] = df["model"].apply(lambda x: MODEL_SIMPLE_NAMES[x])  # type: ignore
    df["is_correct"] = df.ground_truth == df.parsed_response

    if aggregate_formatters:
        avg_n_ans = outputs.map(lambda x: len(x.task_spec.get_data_example_obj().get_options())).average()
        assert avg_n_ans is not None
        g1 = catplot(data=df, x="task_name", y="matches_bias", hue="model", kind="bar", add_line_at=1 / avg_n_ans)
        g2 = catplot(data=df, x="task_name", y="is_correct", hue="model", kind="bar")
    else:
        for formatter in df.formatter_name.unique():
            formatter_df = df[df.formatter_name == formatter]
            avg_n_ans = (
                outputs.filter(lambda x: x.task_spec.formatter_name == formatter)
                .map(lambda x: len(x.task_spec.get_data_example_obj().get_options()))
                .average()
            )
            assert avg_n_ans is not None
            g1 = catplot(
                data=formatter_df, x="task_name", y="matches_bias", hue="model", kind="bar", add_line_at=1 / avg_n_ans
            )
            g2 = catplot(
                data=formatter_df,
                x="task_name",
                y="is_correct",
                hue="model",
                kind="bar",
            )
            g1.fig.suptitle(formatter)
            g2.fig.suptitle(formatter)

    plt.show()


if __name__ == "__main__":
    fire.Fire({"run": run, "plot": plot})
    # plot()
