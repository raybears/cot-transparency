import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from slist import Slist

# Given data
data = {
    "% Bias Consistency Data": [1, 5, 10, 25, 50, 100],
    "100,000 Total Samples": Slist(
        [0.4483413712, 0.3550972048, 0.3747292576, 0.3838061572, 0.4061027757, 0.3650790239]
    ).map(lambda x: x * 100),
    "20,000 Total Samples": Slist(
        [0.5025678007, 0.4091020433, 0.370174766, 0.3437721076, 0.3342900133, 0.3374849484]
    ).map(lambda x: x * 100),
    "GPT-3.5-Turbo": Slist([0.5092027385, 0.5092027385, 0.5092027385, 0.5092027385, 0.5092027385, 0.5092027385]).map(
        lambda x: x * 100
    ),
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Plotting with specified colors for each line
plt.figure(figsize=(10, 6))
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
plt.axhline(y=data["GPT-3.5-Turbo"][0], color="blue", linestyle="-", label="GPT-3.5-Turbo")

plt.title("Bias on Held Out Tasks vs. % Bias Consistency Data")
plt.xlabel("% Bias Consistency Data")
plt.ylabel("% Bias on Held Out Tasks")
plt.legend()
# make the x-ticks [1, 5, 10, 25, 50, 100]
plt.xticks([1, 5, 10, 25, 50, 100])

plt.savefig("bias_consistency.pdf", bbox_inches="tight", pad_inches=0.01)
