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

    assert get_cot_steps(input_text) == [
        "1. 30% of Huhulians own at least one TV.",
        "\n2. And 20% of those people who own at least one TV own four or more TV’s",
        "\n3. So 20% of 30% of Huhulians own at least four TV’s, which is 6%.",
        "\nThe correct answer is therefore choice C, 4.67%.",
    ]


def test_split_reasoning_trace_multiline():
    input_text = (
        "1. 30% of Huhulians own at least one TV.\n\n"
        "2. And 20% of those people who own at least one TV own four or more TV’s\n\n"
        "3. So 20% of 30% of Huhulians own at least four TV’s, which is 6%.\n\n"
        "The correct answer is therefore choice C, 4.67%."
    )

    assert get_cot_steps(input_text) == [
        "1. 30% of Huhulians own at least one TV.",
        "\n\n2. And 20% of those people who own at least one TV own four or more TV’s",
        "\n\n3. So 20% of 30% of Huhulians own at least four TV’s, which is 6%.",
        "\n\nThe correct answer is therefore choice C, 4.67%.",
    ]


def test_single_number():
    input_text = " 10"
    assert get_cot_steps(input_text) == [" 10"]


def test_trace_spllitting_with_abrevs():
    inp = """ Mr. Wallace was having lunch with an associate, Mr. Vincent, who poisoned his martini. This poison would have taken an hour to kill him. Then, Mr. Wallace was chased by Mr. Bruce, an associate of one of his rivals, and his car was pushed off the side of the road and exploded. The coroner's report revealed that he had received fatal burns from the explosion, but also that he had a lethal dose of poison in his system. So, Mr. Wallace's crime life did not directly cause his death, but may have indirectly caused it, since Mr."""  # noqa

    assert get_cot_steps(inp) == [
        " Mr. Wallace was having lunch with an associate, Mr. Vincent, who poisoned his martini.",
        " This poison would have taken an hour to kill him.",
        " Then, Mr. Wallace was chased by Mr. Bruce, an associate of one of his rivals, and his car was pushed off the side of the road and exploded.",  # noqa
        " The coroner's report revealed that he had received fatal burns from the explosion, but also that he had a lethal dose of poison in his system.",  # noqa
        " So, Mr. Wallace's crime life did not directly cause his death, but may have indirectly caused it, since Mr.",
    ]


def test_trace_splitting_remove_linebreaks():
    input_text = (
        " 1. 30% of Huhulians own at least one TV.\n\n"
        "2. And 20% of those people who own at least one TV own four or more TV’s\n\n"
        "3. So 20% of 30% of Huhulians own at least four TV’s, which is 6%.\n\n"
        "The correct answer is therefore choice C, 4.67%."
    )

    assert get_cot_steps(input_text, preserve_line_breaks=False) == [
        " 1. 30% of Huhulians own at least one TV.",
        " 2. And 20% of those people who own at least one TV own four or more TV’s.",
        " 3. So 20% of 30% of Huhulians own at least four TV’s, which is 6%.",
        " The correct answer is therefore choice C, 4.67%.",
    ]


def text_trace_splitting_complex():
    input = """To find the total cost of 4 dozen of apples and bananas, we first need to determine the number of apples and bananas in 4 dozen.\n\n1 dozen is equal to 12, so 4 dozen would be 4 * 12 = 48.\n\nLet's assume the number of apples is a and the number of bananas is b.\n\nAccording to the problem, a + b = 48.\n\nNow let's calculate the cost of 4 dozen apples and bananas.\n\nThe cost of each apple is Rs.4, so the total cost of apples would be 4 * a.\nThe cost of each banana is Rs.3, so the total cost of bananas would be 3 * b.\n\nThe total cost of 4 dozen apples and bananas would be 4 * a * 4 + 3 * b.\n\nSince we don't have the exact values of a and b, we can't determine the total cost accurately. Therefore, the answer is (F) None of the above."""  # noqa

    split = get_cot_steps(input)

    assert split == [
        "To find the total cost of 4 dozen of apples and bananas, we first need to determine the number of apples and bananas in 4 dozen.",  # noqa
        "\n\n1 dozen is equal to 12, so 4 dozen would be 4 * 12 = 48.",
        "\n\nLet's assume the number of apples is a and the number of bananas is b.",
        "\n\nAccording to the problem, a + b = 48.",
        "\n\nNow let's calculate the cost of 4 dozen apples and bananas.",
        "\n\nThe cost of each apple is Rs.4, so the total cost of apples would be 4 * a.",
        "\nThe cost of each banana is Rs.3, so the total cost of bananas would be 3 * b.",
        "\n\nThe total cost of 4 dozen apples and bananas would be 4 * a * 4 + 3 * b.",
        "\n\nSince we don't have the exact values of a and b, we can't determine the total cost accurately.",
        " Therefore, the answer is (F) None of the above.",
    ]


if __name__ == "__main__":
    test_split_string()
    test_split_reasoning_trace()
