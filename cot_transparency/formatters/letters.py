from string import ascii_uppercase


def answer_idx_to_letter_bracket(idx: int) -> str:
    return f"({ans_map_to_let[idx]})"


def index_to_letter(idx: int) -> str:
    return ans_map_to_let[idx]


ans_map_to_let: dict[int, str] = {k: v for k, v in zip(range(26), ascii_uppercase)}
