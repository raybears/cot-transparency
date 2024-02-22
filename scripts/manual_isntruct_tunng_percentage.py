import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from slist import Slist


# Set the font scale
# sns.set_context("notebook", font_scale=1.2)

# Given data
data = {
    "% Bias Consistency Data": [1, 2, 5, 10, 25, 50, 100],
    "100,000 Total Samples": Slist(
        # 16.36%	27.10%	49.74%	40.29%	34.90%	29.28%	34.10%
        [0.164, 0.2710, 0.497, 0.403, 0.349, 0.293, 0.341]
    ).map(lambda x: x * 100),
    "20,000 Total Samples": Slist(
        # 11.4%	20.87%, 38.5%	44.1%	53.0%	51.3%	43.5%
        [0.114, 0.2087, 0.385, 0.441, 0.53, 0.513, 0.435]
    ).map(lambda x: x * 100),
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Plotting with specified colors for each line
# Set the figure size
ax = plt.subplot()
sns.lineplot(
    x="% Bias Consistency Data",
    y="100,000 Total Samples",
    data=df,
    marker="o",
    color="green",
    label="100,000 Total Samples",
)
sns.lineplot(
    x="% Bias Consistency Data",
    y="20,000 Total Samples",
    data=df,
    marker="o",
    color="orange",
    label="20,000 Total Samples",
)


# For the GPT-3.5-Turbo, since it is a constant line, we can use plt.axhline
# plt.axhline(y=data["GPT-3.5-Turbo"][0], color="blue", linestyle="-", label="GPT-3.5-Turbo")

# plt.title("Bias on Held Out Tasks vs. % Bias Consistency Data")
plt.xlabel("% Bias Consistency Data (rest is instruct-tuning data)")
plt.ylabel("% Bias Relative Decrease")
# legend on the top right
plt.legend(loc="lower right")

plt.xscale("log")
# show only this on
ax.set_xticks([1, 2, 5, 10, 25, 50, 100])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())  # to avoid scientific notation

plt.ylim(0, 60)

plt.savefig("instruction_prop_impact_new.pdf", bbox_inches="tight", pad_inches=0.01)
