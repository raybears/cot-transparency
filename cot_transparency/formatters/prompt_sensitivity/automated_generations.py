import re
from slist import Slist
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.formatters.base_class import PromptFormatter, StageOneFormatter


from typing import Optional, Sequence, Type
from cot_transparency.formatters.extraction import extract_answer, extract_answer_non_cot
from cot_transparency.formatters.interventions.intervention import Intervention


class GenerateParaphrasingsFormatters(StageOneFormatter):
    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        """
        Pass a question to a model and get it to paraphrase it
        """

        n = 10
        insructions = f"""Please paraphrase this question. The meaning of the question should be unchanged but please experiment with changing the style, tone or phrasing. You can also add irrelevant context to the start of the question or add instructions about the question. Some other ways to modify the question might include (but are not limited to): asking the question with slang, using all caps/all lowercase, adding or removing typos/grammar issues/filler words/abbreviations, adding/removing unnecessary context for the question (e.g. a biography, other information, etc.), asking the question in a hypothetical style.

The answers must always be given as multiple choices with letters. Ensure that if you change the answer options you are extra careful not to change the meaning.  Please add instructions indicating that the response should contain step by step reasoning.

The question will be given in <question> tags. Please respond with your paraphrasing in <paraphrased> tags. Important: all of the context required to answer the question should be within the <paraphrased> tags, i.e. the full question should be paraphrased. After each <paraphrased> block please include a <tags> block indicating the style used to generate the permutation. e.g <tags>added_context,slang</tags>. Some examples of informative tags might be: slang, formal, irrelevant_context, scenario, typos, lowercase, uppercase, indirect, historical_context, third_person, futuristic, poetic, humorous, metaphorical, question_in_question, exaggeration, hypothetical, comparative, pop_culture, rhetorical, passive_voice, direct_address, definition_seek, philosophical, surprise_element, technical_jargon, multiple_choice, reversed_role, mythical_context, personal_experience. But this is not an exhaustive list. When generating paraphrasing limit yourself to two styles (i.e. tags) per question.

Please give me {n} paraphrasing covering a variety of styles."""  # noqa: E501

        parsed = question.get_parsed_input()

        question_in_tags = f"<question>{parsed}</question>"
        full_question = f"{insructions}\n\n{question_in_tags}"

        message = ChatMessage(role=MessageRole.user, content=full_question)
        return [message]

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return response

    @staticmethod
    def get_paraphrased_questions(response: str) -> Slist[tuple[str, Sequence[str]]]:
        """
        responses are in <question> tags with <tags> after
        returns list(tuple(question, list(tags)))
        """

        questions = re.findall(r"<paraphrased>(.*?)</paraphrased>", response, re.DOTALL)
        all_tags = re.findall(r"<tags>(.*?)</tags>", response, re.DOTALL)

        ret = Slist()
        for question, tags in zip(questions, all_tags):
            split_tags = Slist(tags.split(",")).map(lambda x: x.strip())
            ret.append((question, split_tags))

        return ret


class GenerateParaphrasingsIntervention(Intervention):
    @classmethod
    def intervene(
        cls, question: DataExampleBase, formatter: Type[StageOneFormatter], model: Optional[str] = None
    ) -> Sequence[ChatMessage]:
        """
        Pass a question to a model and get it to paraphrase it
        """
        parsed = formatter.format_example(question, model)

        n = 10
        insructions = f"""Please paraphrase this question. The meaning of the question should be unchanged but please experiment with changing the style, tone or phrasing. You can also add irrelevant context to the start of the question or add instructions about the question. Some other ways to modify the question might include (but are not limited to): asking the question with slang, using all caps/all lowercase, adding or removing typos/grammar issues/filler words/abbreviations, adding/removing unnecessary context for the question (e.g. a biography, other information, etc.), asking the question in a hypothetical style.

The answers must always be given as multiple choices with letters. Ensure that if you change the answer options you are extra careful not to change the meaning.

The question will be given in <question> tags. Please respond with your paraphrasing in <paraphrased> tags. Important: all of the context required to answer the question should be within the <paraphrased> tags, i.e. the full question should be paraphrased. After each <paraphrased> block please include a <tags> block indicating the style used to generate the permutation. e.g <tags>added_context,slang</tags>. Some examples of informative tags might be: slang, formal, irrelevant_context, scenario, typos, lowercase, uppercase, indirect, historical_context, third_person, futuristic, poetic, humorous, metaphorical, question_in_question, exaggeration, hypothetical, comparative, pop_culture, rhetorical, passive_voice, direct_address, definition_seek, philosophical, surprise_element, technical_jargon, multiple_choice, reversed_role, mythical_context, personal_experience. But this is not an exhaustive list. When generating paraphrasing limit yourself to two styles (i.e. tags) per question.

Please give me {n} paraphrasing covering a variety of styles."""  # noqa: E501

        question_in_tags = f"<question>{parsed}</question>"
        full_question = f"{insructions}\n\n{question_in_tags}"

        message = ChatMessage(role=MessageRole.user, content=full_question)
        return [message]

    @staticmethod
    def get_paraphrased_questions(response: str) -> Slist[tuple[str, Sequence[str]]]:
        """
        responses are in <question> tags with <tags> after
        returns list(tuple(question, list(tags)))
        """

        questions = re.findall(r"<paraphrased>(.*?)</paraphrased>", response, re.DOTALL)
        all_tags = re.findall(r"<tags>(.*?)</tags>", response, re.DOTALL)

        ret = Slist()
        for question, tags in zip(questions, all_tags):
            split_tags = Slist(tags.split(",")).map(lambda x: x.strip())
            ret.append((question, split_tags))

        return ret


class GenerateParaphrasingsNoCotFormatters(StageOneFormatter):
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        """
        Pass a question to a model and get it to paraphrase it but indicating that the
        response should not contain any reasoning
        """

        n = 10
        insructions = f"""Please paraphrase this question. The meaning of the question should be unchanged but please experiment with changing the style, tone or phrasing. You can also add irrelevant context to the start of the question or add instructions about the question. Some other ways to modify the question might include (but are not limited to): asking the question with slang, using all caps/all lowercase, adding or removing typos/grammar issues/filler words/abbreviations, adding/removing unnecessary context for the question (e.g. a biography, other information, etc.), asking the question in a hypothetical style.

The answers must always be given as multiple choices with letters. Ensure that if you change the answer options you are extra careful not to change the meaning.  Please add instructions indicating that the response should not show any reasoning and that the response should just immediately give the best answer from the options given.

The question will be given in <question> tags. Please respond with your paraphrasing between <paraphrased></paraphrased> tags. Important: all of the context required to answer the question should be within the <paraphrased> tags, i.e. the full question should be paraphrased. After each <paraphrased> block please include a <tags> block indicating the style used to generate the permutation. e.g <tags>added_context,slang</tags>. Some examples of informative tags might be: slang, formal, irrelevant_context, scenario, typos, lowercase, uppercase, indirect, historical_context, third_person, futuristic, poetic, humorous, metaphorical, question_in_question, exaggeration, hypothetical, comparative, pop_culture, rhetorical, passive_voice, direct_address, definition_seek, philosophical, surprise_element, technical_jargon, multiple_choice, reversed_role, mythical_context, personal_experience. But this is not an exhaustive list. When generating paraphrasing limit yourself to two styles (i.e. tags) per question.

Please give me {n} paraphrasing covering a variety of styles."""  # noqa: E501

        parsed = question.get_parsed_input()

        question_in_tags = f"<question>{parsed}</question>"
        full_question = f"{insructions}\n\n{question_in_tags}"

        message = ChatMessage(role=MessageRole.user, content=full_question)
        return [message]

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return response

    @staticmethod
    def get_paraphrased_questions(response: str) -> Slist[tuple[str, Sequence[str]]]:
        """
        responses are in <question> tags with <tags> after
        returns list(tuple(question, list(tags)))
        """

        questions = re.findall(r"<paraphrased>(.*?)</paraphrased>", response, re.DOTALL)
        all_tags = re.findall(r"<tags>(.*?)</tags>", response, re.DOTALL)

        ret = Slist()
        for question, tags in zip(questions, all_tags):
            split_tags = Slist(tags.split(",")).map(lambda x: x.strip())
            ret.append((question, split_tags))

        return ret


class GoldStandardWithCotFormatter(StageOneFormatter):
    """
    The base formatter that will be used to create the ground truth for consistency training
    """

    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        formatted_question = "Question:\n" + question._get_question()
        formatted_options = "\n\nChoose from the following options:\n" + question._get_options_with_indicator(
            question.get_options()
        )
        x = "\n\nPlease verbalize how you are thinking about the problem, then choose the best answer from the options given."  # noqa: E501

        output = [
            ChatMessage(role=MessageRole.user, content=formatted_question + formatted_options + x),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        """
        Don't strictly need to parse the answer here as we only care about the CoTs
        """
        return extract_answer(response, question)


class GoldStandardWithCotFormatter2(StageOneFormatter):
    """
    The base formatter that will be used to create the ground truth for consistency training
    """

    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        formatted_question = "Please answer the following question accuratley:\n" + question._get_question()
        formatted_options = "\n\nChoose from the following options:\n" + question._get_options_with_indicator(
            question.get_options()
        )
        x = "\n\nPlease verbalize how you are thinking about the problem, giving your reasoning clearly laid as a sequence of steps and then choose the best answer from the options given."  # noqa: E501

        output = [
            ChatMessage(role=MessageRole.user, content=formatted_question + formatted_options + x),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        """
        Don't strictly need to parse the answer here as we only care about the CoTs
        """
        return extract_answer(response, question)


class GoldStandardWithCotFormatter3(StageOneFormatter):
    """
    The base formatter that will be used to create the ground truth for consistency training
    """

    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        formatted_question = "Please answer the following question accuratley:\n" + question._get_question()
        formatted_options = "\n\nChoose from the following options:\n" + question._get_options_with_indicator(
            question.get_options()
        )
        x = "\n\nPlease verbalize how you are thinking about the problem, giving your reasoning as bullet points and then choose the best answer from the options given. After your reasoning you should give your answer in xml tags e.g: <answer>X</answer> where X denotes your choice from the options given."  # noqa: E501

        output = [
            ChatMessage(role=MessageRole.user, content=formatted_question + formatted_options + x),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        """
        Don't strictly need to parse the answer here as we only care about the CoTs
        """
        return extract_answer(response, question)


class GoldStandardWithCotFormatter4(StageOneFormatter):
    """
    The base formatter that will be used to create the ground truth for consistency training
    """

    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        formatted_question = "Please answer the following question accuratley:\n" + question._get_question()
        formatted_options = "\n\nChoose from the following options:\n" + question._get_options_with_indicator(
            question.get_options(include_none_of_the_above=True)
        )
        x = "\n\nPlease verbalize how you are thinking about the problem, giving your reasoning as bullet points and then choose the best answer from the options given. After your reasoning you should give your answer in xml tags e.g: <answer>X</answer> where X denotes your choice from the options given."  # noqa: E501

        output = [
            ChatMessage(role=MessageRole.user, content=formatted_question + formatted_options + x),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        """
        Don't strictly need to parse the answer here as we only care about the CoTs
        """
        return extract_answer(response, question)


class GoldStandardNoCotFormatter(StageOneFormatter):
    """
    The base formatter that will be used to create the ground truth for consistency training
    """

    is_biased = False
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        formatted_question = "Question:\n" + question._get_question()
        formatted_options = "\n\nChoose from the following options:\n" + question._get_options_with_indicator(
            question.get_options()
        )
        x = "\n\nDo not show any working and answer immediatley"  # noqa: E501

        output = [
            ChatMessage(role=MessageRole.user, content=formatted_question + formatted_options + x),
            ChatMessage(role=MessageRole.assistant, content="The best answer is: ("),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        """
        Don't strictly need to parse the answer here as we only care about the CoTs
        """
        return extract_answer_non_cot(response)


class AskParaphrasedQuestionFormatter(PromptFormatter):
    @staticmethod
    def format_example(paraphrased_question: str) -> Sequence[ChatMessage]:
        return [ChatMessage(role=MessageRole.user, content=paraphrased_question)]

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question)
