from matplotlib import pyplot as plt
import pandas as pd
from scripts.utils.plots import catplot

if __name__ == "__main__":
    # Make a bar plot of
    x_names = [
        "GPT-3.5-Turbo",
        "Non-COT majority",
        "Balanced samples\n(Ours)",
    ]
    y_values = [
        52.7,
        42.0,
        34.0,
    ]
    # make a df
    df = pd.DataFrame(
        {
            "x": x_names,
            "y": y_values,
        }
    )
    fig = catplot(
        data=df,
        x="x",
        y="y",
        # width=10,
        # title="Accuracy of GPT-3.5-Turbo- vs Non-COT vs 50-50 COT",
        # xlabel="Model",
        # save_path="plots/accuracy_of_gpt_3_5_turbo_vs_non_cot_vs_50_50_cot.png",
    )
    # show
    # add labels
    # plt.xlabel("Model")
    # plt.ylabel("% Answer Matching Bias")
    # make the title ylabel
    plt.title("% of Answers Matching Bias")
    # plt.show()
    # save
    # plt.savefig("we_need_cot.png")
    # remove x and y axis labels
    plt.xlabel(None)
    plt.ylabel(None)

    # save pdf
    plt.savefig("we_need_cot.pdf", bbox_inches="tight", pad_inches=0.01)
