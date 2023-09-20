import pandas as pd

from cot_transparency.data_models.models import ChatMessage, MessageRole


def sample_few_shot(sample_size=5) -> list[ChatMessage]:
    df = pd.read_csv(
        "./cot_transparency/formatters/transparency/interventions/follow-mistakes-john-level3-2_filter_mistakes.csv"
    )
    mistake_pos_count = df["mistake_added_at"].nunique()
    group_sample_n = 1 if mistake_pos_count >= sample_size else (sample_size % mistake_pos_count) + 1

    # sample rows uniformly for different mistake_added_at
    dfs = [group.sample(n=group_sample_n) for _, group in df.groupby("mistake_added_at")]
    df = pd.concat(dfs)

    # randomly sample
    df = df.sample(frac=1, random_state=None).reset_index(drop=True)
    df = df.sort_values(by="aoc_difference", ascending=False)
    df = df.head(sample_size)

    few_shot_msgs = []
    for _, row in df.iterrows():
        few_shot_msgs.append(ChatMessage(role=MessageRole.user, content=str(row["question"])))
        few_shot_msgs.append(ChatMessage(role=MessageRole.assistant, content=str(row["modified_cot"])))

    return few_shot_msgs
