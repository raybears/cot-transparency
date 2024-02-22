from matplotlib import pyplot as plt
import pandas as pd
from scripts.utils.plots import catplot

if __name__ == "__main__":
    # Make a bar plot of
    x_names = [
        "Control",
        "Non-CoT",
        "BCT",
    ]
    y_values = [
        44.7,
        36.4,
        33.7,
    ]
    # sent font to times new roman
    plt.rcParams["font.family"] = "Times New Roman"
    # make a df
    df = pd.DataFrame(
        {
            "x": x_names,
            "y": y_values,
        }
    )
    # fig with smaller width
    fig = catplot(
        data=df,
        x="x",
        y="y",
        width=4,
        # title="Accuracy of GPT-3.5-Turbo- vs Non-COT vs 50-50 COT",
        # xlabel="Model",
        # save_path="plots/accuracy_of_gpt_3_5_turbo_vs_non_cot_vs_50_50_cot.png",
    )
    red_dotted_value = 49.7
    # make a red dotted line with legend label
    plt.axhline(y=red_dotted_value, color="r", linestyle="--", label="GPT-3.5\nwith biasing\n prompt")
    black_dotted_value = 9.8
    # make a black dotted line with legend label "GPT-3.5 unbiased baseline"
    plt.axhline(y=black_dotted_value, color="k", linestyle="--", label="GPT-3.5\nwithout biasing\n prompt")

    # legend in middle, opacity 0.5, small
    plt.legend(loc="lower right", framealpha=0.9, fontsize="small")

    # y-axis label of % Answers Matching Bias
    plt.ylabel("% Answers Matching Bias")

    # red dottted li
    # show
    # add labels
    # plt.xlabel("Model")
    # plt.ylabel("% Answer Matching Bias")
    # make the title ylabel
    # plt.title("% Bias across tasks")
    # plt.show()
    # save
    # plt.savefig("we_need_cot.png")
    # remove x and y axis labels
    plt.xlabel(None)  # type: ignore
    # Make the max y value 50
    plt.ylim(0, 50)
    # Write the actual values
    for i, v in enumerate(y_values):
        plt.text(i, v + 1, str(v), ha="center")

    # save pdf
    plt.savefig("we_need_cot.pdf", bbox_inches="tight", pad_inches=0.01)
