from scripts.biased_wrong_ans import bad_cot_extraction


def test_bad_cot_extraction():
    test = """First, we start by facing forward.
The best answer is: (B) False.
"""
    assert bad_cot_extraction(test) == """First, we start by facing forward."""

    none_test = """First, we start by facing forward."""
    assert bad_cot_extraction(none_test) is None
