from cot_transparency.formatting.emoji.emoji_formatter import extract_multiple_choices


def test_extract_multiple_choices():
    test_case = (
        "Q: Which of the following is a humorous edit of this artist or movie name: 'empire of the ants'?\n\n"
        "Answer choices:\n(A) empire of the pants\n(B) empiqe of the ants\n(C) empire of tihe ants\n"
        "(D) empire of the antts"
    )
    expected = ["empire of the pants", "empiqe of the ants", "empire of tihe ants", "empire of the antts"]
    assert extract_multiple_choices(test_case) == expected
