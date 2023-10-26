from cot_transparency.formatters.wildcard import match_wildcard_formatters


def test_match_wildcard_formatters():
    assert match_wildcard_formatters(["ZeroShotUnbiasedFormatte*"]) == [
        "ZeroShotUnbiasedFormatter"
    ]
