import fire
from cot_transparency.formatters import name_to_formatter
from cot_transparency.model_apis import Prompt, convert_to_strict_messages
from tests.test_prompt_formatter import EMPIRE_OF_PANTS_EXAMPLE


def main(formatter_name: str):
    formatter = name_to_formatter(formatter_name)

    example = formatter.format_example(EMPIRE_OF_PANTS_EXAMPLE)  # type: ignore

    try:
        print("Anthropic layout")
        print(Prompt(messages=convert_to_strict_messages(example, "claude-v1")).convert_to_anthropic_str())
    except ValueError:
        pass

    print("\nOpenAI Completion layout")
    print(Prompt(messages=convert_to_strict_messages(example, "text-davinci-003")).convert_to_completion_str())

    try:
        print("\nOpenAI Chat layout")
        print(convert_to_strict_messages(example, "gpt-4"))
    except ValueError:
        pass


if __name__ == "__main__":
    fire.Fire(main)
