from string import ascii_uppercase
from cot_transparency.data_models.messages import ChatMessage
from cot_transparency.formatters.util import load_few_shots
import pytest

# parameterize test over paths ./data/ethan_few_shot.txt and ./data/gpt4_generated_few_shot.txt


@pytest.mark.parametrize("path", ["./data/ethan_few_shot.txt", "./data/gpt4_generated_few_shot.txt"])
def test_generic_few_shot_have_answer(path: str):
    few_shots: list[tuple[ChatMessage, ChatMessage]] = load_few_shots(path)

    for _, assistant_msg in few_shots:
        assert assistant_msg.content[-3] in ascii_uppercase

    pass
