from scripts.demo_formatter import main

COT_FORMATTERS = [
    "CotPromptSenFormatter_LETTERS_SHORT_SELECT_PAREN_NEWLINE",
    "CotPromptSenFormatter_LETTERS_PLEASE_SELECT_DOT_SENTENCE",
    "CotPromptSenFormatter_NUMBERS_SHORT_OPTIONS_PAREN_NEWLINE",
    "CotPromptSenFormatter_NUMBERS_NONE_SELECT_DOT_SENTENCE",
    "CotPromptSenFormatter_NUMBERS_TAG_OPTIONS_DOT_NEWLINE",
    "CotPromptSenFormatter_NUMBERS_PLEASE_NONE_DOT_SENTENCE",
    "CotPromptSenFormatter_LETTERS_SHORT_OPTIONS_PAREN_NEWLINE",
    "CotPromptSenFormatter_NUMBERS_SHORT_ANS_CHOICES_PAREN_NEWLINE",
    "CotPromptSenFormatter_LETTERS_PLEASE_OPTIONS_DOT_NEWLINE",
    "CotPromptSenFormatter_NUMBERS_NONE_ANS_CHOICES_PAREN_SENTENCE",
]

for formatter in COT_FORMATTERS:
    print()
    print("Running formatter: ", formatter, "-----------------------")
    main(formatter_name=formatter, layout="gpt")
