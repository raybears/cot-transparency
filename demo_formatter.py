from typing import Optional, Type
import fire
from cot_transparency.formatters import name_to_formatter
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.model_apis import Prompt
from tests.prompt_formatting.test_prompt_formatter import EMPIRE_OF_PANTS_EXAMPLE
from stage_one import get_valid_stage1_interventions


def main(
    formatter_name: Optional[str] = None,
    intervention_name: Optional[str] = None,
    model="gpt-4",
    layout: str = "anthropic",
):
    try:
        formatter: Type[StageOneFormatter]
        formatter = name_to_formatter(formatter_name)  # type: ignore
    except KeyError:
        print("Formatter name not found")
        print("Available formatters:")
        for formatter_name in StageOneFormatter.all_formatters():
            print(formatter_name)
        return

    if intervention_name:
        Intervention = get_valid_stage1_interventions([intervention_name])[0]
        example = Intervention.intervene(question=EMPIRE_OF_PANTS_EXAMPLE, formatter=formatter, model=model)
    else:
        example = formatter.format_example(EMPIRE_OF_PANTS_EXAMPLE, model=model)  # type: ignore

    prompt = Prompt(messages=example)

    if layout == "anthropic":
        print("Anthropic layout")
        print(prompt.convert_to_anthropic_str())

    elif layout == "openai-completion":
        print("\nOpenAI Completion layout")
        print(prompt.convert_to_completion_str())

    else:
        print("\nOpenAI Chat layout")
        print(prompt.convert_to_openai_chat())


if __name__ == "__main__":
    fire.Fire(main)
