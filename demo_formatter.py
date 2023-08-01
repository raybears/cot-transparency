import fire
from cot_transparency.formatters import name_to_formatter
from cot_transparency.model_apis import convert_to_completion_str, convert_to_strict_messages
from tests.test_prompt_formatter import EMPIRE_OF_PANTS_EXAMPLE


def main(formatter_name: str):
    formatter = name_to_formatter(formatter_name)

    example = formatter.format_example(EMPIRE_OF_PANTS_EXAMPLE)  # type: ignore

    print("Anthropic layout")
    print(convert_to_completion_str(convert_to_strict_messages(example, "claude-v1")))

    print("\nOpenAI Completion layout")
    print(convert_to_completion_str(convert_to_strict_messages(example, "text-davinci-003")))

    print("\nOpenAI Chat layout")
    print(convert_to_strict_messages(example, "gpt-4"))


if __name__ == "__main__":
    fire.Fire(main)
