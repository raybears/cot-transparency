import pandas as pd
from scripts.prompt_sensitivity import fleiss_kappa_on_group


def test_fleiss_kappa_on_total_disagreement():
    # 1. Mock data
    data = {
        "model": ["model1", "model1", "model1", "model1"],
        "task_hash": ["task1", "task1", "task2", "task2"],
        "formatter_name": ["PromptFormat1", "PromptFormat2", "PromptFormat1", "PromptFormat2"],
        "parsed_response": [1, 2, 2, 1],
    }

    group = pd.DataFrame(data)
    # 2. Expected output (this is a mocked value; in a real test, you'd compute this expected value separately)
    expected_output = group.copy()
    expected_output["fleiss_kappa"] = float(-1)

    # 3. Call your function
    output = group.groupby("model").apply(fleiss_kappa_on_group).reset_index(drop=True)

    # 4. Assert if the output is as expected
    pd.testing.assert_frame_equal(output, expected_output)


def test_fleiss_kappa_on_total_greement():
    # 1. Mock data
    data = {
        "model": ["model1", "model1", "model1", "model1"],
        "task_hash": ["task1", "task1", "task2", "task2"],
        "formatter_name": ["PromptFormat1", "PromptFormat2", "PromptFormat1", "PromptFormat2"],
        "parsed_response": [1, 1, 2, 2],
    }

    group = pd.DataFrame(data)
    # 2. Expected output (this is a mocked value; in a real test, you'd compute this expected value separately)
    expected_output = group.copy()
    expected_output["fleiss_kappa"] = float(1)

    # 3. Call your function
    output = group.groupby("model").apply(fleiss_kappa_on_group).reset_index(drop=True)

    # 4. Assert if the output is as expected
    pd.testing.assert_frame_equal(output, expected_output)