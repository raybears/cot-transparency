from scripts.biased_wrong_ans import cot_extraction


def test_cot_extraction():
    test = """First, we start by facing forward.
The best answer is: (B) False.
"""
    assert cot_extraction(test) == """First, we start by facing forward."""

    none_test = """First, we start by facing forward."""
    assert cot_extraction(none_test) is None
