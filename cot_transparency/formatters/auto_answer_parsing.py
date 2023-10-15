from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.formatters.transparency.util import StageTwoFormatter
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel


class AnswerParsingExample(BaseModel):
    options: str
    response: str
    parsed_answer: str


@lru_cache(maxsize=1)
def read_few_shot_examples():
    inp_path = "data/training_prompt_sen/answer_parsing_few_shots_filtered.jsonl"
    slist = read_jsonl_file_into_basemodel(Path(inp_path), AnswerParsingExample)
    return slist


class GetAnswerGivenFormatter(StageTwoFormatter):
    is_intermediate = False

    @staticmethod
    def format_example(model_response: str, original_question: DataExampleBase, model: str) -> list[ChatMessage]:
        """
        This basically uses an LM to "find" the answer given in the response, this does not need to
        use the same model that generated the response as we assume that we could identify the answer
        if we were to read the response but we want to automate this process.
        """

        start = """I have a collection of responses to multiple choice questions, please help me parse out the answers that were given in the response. I will first give you the options that were included in the original question enclosed in <options> tags and then the response to parse in <text> tags. You can respond in two ways, either with the single capital letter indicating the parsed response or with the string "none" if the response either chooses multiple options or does not choose any options at all.\n\nHere are some examples:"""  # noqa

        few_shot_examples = read_few_shot_examples()

        examples = ""
        for example in few_shot_examples:
            examples += (
                f"\n<example>\n<options>\n{example.options}\n<options>\n<text>{example.response}\n</text>"
            )
            examples += "\n\nThe answer given in the <text> is ({example.answer})</example>"

        original_options = original_question._get_options_with_indicator(original_question.get_options())
        actual_question = "<options>\n" + original_options + "\n<options>\n<text>" + model_response + "\n</text>"

        user_message = start + examples + "\n\n" + actual_question

        assistant_message = "The answer given in the <text> is ("
        output = [
            ChatMessage(role=MessageRole.user, content=user_message),
            ChatMessage(role=MessageRole.assistant, content=assistant_message),
        ]
        return output
