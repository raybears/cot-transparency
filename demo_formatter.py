from typing import Optional, Type
import fire
from cot_transparency.formatters import name_to_formatter
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.model_apis import Prompt
from tests.prompt_formatting.test_prompt_formatter import EMPIRE_OF_PANTS_EXAMPLE
from stage_one import get_valid_stage1_interventions


def main(formatter_name: str, intervention_name: Optional[str] = None, model="gpt-4"):
    formatter: Type[StageOneFormatter]
    formatter = name_to_formatter(formatter_name)  # type: ignore

    if intervention_name:
        Intervention = get_valid_stage1_interventions([intervention_name])[0]
        example = Intervention.intervene(question=EMPIRE_OF_PANTS_EXAMPLE, formatter=formatter, model=model)
    else:
        example = formatter.format_example(EMPIRE_OF_PANTS_EXAMPLE, model=model)  # type: ignore

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
