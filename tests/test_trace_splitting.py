from cot_transparency.formatters.transparency.trace_manipulation import (
    get_cot_steps,
    split_string,
)


def test_split_string():
    text1 = "\nHello\nWorld"
    assert split_string(text1) == ["\nHello", "\nWorld"]

    text2 = "This is a sentence. Here's another one!\nThis is a new line. It has two sentences."
    assert split_string(text2) == [
        "This is a sentence.",
        " Here's another one!",
        "\nThis is a new line.",
        " It has two sentences.",
    ]

    text3 = "No newlines in this string."
    assert split_string(text3) == ["No newlines in this string."]


def test_split_reasoning_trace():
    input_text = (
        "30% of Huhulians own at least one TV. And 20% of those people who own at"
        " least one TV own four or more TV’s. So 20% of 30% of Huhulians own at "
        "least four TV’s, which is 6%. The correct answer is therefore choice C, 4.67%."
    )

    assert get_cot_steps(input_text) == [
        "30% of Huhulians own at least one TV.",
        " And 20% of those people who own at least one TV own four or more TV’s.",
        " So 20% of 30% of Huhulians own at least four TV’s, which is 6%.",
        " The correct answer is therefore choice C, 4.67%.",
    ]

    input_text = (
        "1. 30% of Huhulians own at least one TV.\n"
        "2. And 20% of those people who own at least one TV own four or more TV’s\n"
        "3. So 20% of 30% of Huhulians own at least four TV’s, which is 6%.\n"
        "The correct answer is therefore choice C, 4.67%."
    )
    print(split_string(input_text))

    assert get_cot_steps(input_text) == [
        "1. 30% of Huhulians own at least one TV.",
        "\n2. And 20% of those people who own at least one TV own four or more TV’s",
        "\n3. So 20% of 30% of Huhulians own at least four TV’s, which is 6%.",
        "\nThe correct answer is therefore choice C, 4.67%.",
    ]


if __name__ == "__main__":
    test_split_string()
    test_split_reasoning_trace()
