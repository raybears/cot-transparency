import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from slist import Slist


# Set the font scale
# sns.set_context("notebook", font_scale=1.2)

# Given data
data = {
    "% BCT Data": [1, 2, 5, 10, 25, 50, 100],
    "100,000 Total Samples": Slist(
        # 43.19129643	37.9	32.7059695	36.09245015	38.46853025	37.30911524	342
        [0.4319129643, 0.379, 0.327059695, 0.3609245015, 0.3846853025, 0.3730911524, 0.342]
    ).map(lambda x: x * 100),
    "20,000 Total Samples": Slist(
        # 49.45491908	43.19361214	38.57130229	35.67203023	33.61659571	35.91	32.88674981
        [0.4945491908, 0.4319361214, 0.3857130229, 0.3567203023, 0.3361659571, 0.3591, 0.3288674981]
    ).map(lambda x: x * 100),
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Plotting with specified colors for each line
# Set the figure size
ax = plt.subplot()
sns.lineplot(
    x="% BCT Data",
    y="100,000 Total Samples",
    data=df,
    marker="o",
    color="green",
    label="100,000 Total Samples",
)
sns.lineplot(
    x="% BCT Data",
    y="20,000 Total Samples",
    data=df,
    marker="o",
    color="orange",
    label="20,000 Total Samples",
)


# For the GPT-3.5-Turbo, since it is a constant line, we can use plt.axhline
# plt.axhline(y=data["GPT-3.5-Turbo"][0], color="blue", linestyle="-", label="GPT-3.5-Turbo")

# plt.title("Bias on Held Out Tasks vs. % Bias Consistency Data")
plt.xlabel("% BCT Data (rest is instruct-tuning data)")
plt.ylabel("% Answers matching bias")

plt.xscale("log")
# show only this on
ax.set_xticks([1, 2, 5, 10, 25, 50, 100])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())  # to avoid scientific notation

red_dotted_value = 51.4
# make a red dotted line with legend label
plt.axhline(y=red_dotted_value, color="r", linestyle="--", label="GPT-3.5 with biasing prompt")
black_dotted_value = 9.5
# make a black dotted line with legend label "GPT-3.5 unbiased baseline"
plt.axhline(y=black_dotted_value, color="k", linestyle="--", label="GPT-3.5 without biasing prompt")
plt.ylim(0, 60)
# legend opaque
plt.legend(loc="lower right", framealpha=1.00)

plt.savefig("instruction_prop_impact_new.pdf", bbox_inches="tight", pad_inches=0.01)
