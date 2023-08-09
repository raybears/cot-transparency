import fire
from cot_transparency.formatters import name_to_formatter
from cot_transparency.model_apis import Prompt
from tests.prompt_formatting.test_prompt_formatter import EMPIRE_OF_PANTS_EXAMPLE


def main(formatter_name: str):
    formatter = name_to_formatter(formatter_name)

    example = formatter.format_example(EMPIRE_OF_PANTS_EXAMPLE)  # type: ignore
    prompt = Prompt(messages=example)

    try:
        print("Anthropic layout")
        print(prompt.convert_to_anthropic_str())
    except ValueError:
        pass

    print("\nOpenAI Completion layout")
    print(prompt.convert_to_completion_str())

    try:
        print("\nOpenAI Chat layout")
        print(prompt.convert_to_openai_chat())
    except ValueError:
        pass


if __name__ == "__main__":
    fire.Fire(main)
