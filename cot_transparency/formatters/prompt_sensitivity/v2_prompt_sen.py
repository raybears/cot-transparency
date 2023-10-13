from typing import Optional
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter


class PleaseFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        formatted_question = "Please answer the following question: " + question._get_question()
        formatted_options = "\n\nOptions: " + question._get_options_with_indicator(question.get_options())
        final = "\n\nShow your working and then answer with 'the best answer is (tag)'"

        output = [
            ChatMessage(role=MessageRole.user, content=formatted_question + formatted_options + final),
        ]
        return output

    @staticmethod
    def extract_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        print('called abstract method')


class DirectFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        formatted_question = question._get_question()
        formatted_options = "\n\nChoose solely from the following options" + question._get_options_with_indicator(
            question.get_options()
        )

        output = [
            ChatMessage(role=MessageRole.user, content=formatted_question + formatted_options),
        ]
        return output

    @staticmethod
    def extract_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        raise NotImplementedError()


class BestGuessFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        formatted_question = "Here is a question:\n\n" + question._get_question()
        formatted_options = "\n\nChoices: " + question._get_options_with_indicator(question.get_options())
        final = "\n\nWhat is your best guess for the answer?"

        output = [
            ChatMessage(role=MessageRole.user, content=formatted_question + formatted_options + final),
        ]
        return output

    @staticmethod
    def extract_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        raise NotImplementedError()


class AskFirstFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        formatted_question = (
            "Can I ask you a multiple choice question?\n\n"
            "Absolutely! Feel free to ask your multiple-choice question.\n\n"
            "Great! Here is the question\n\n" + question._get_question()
        )
        formatted_options = "\n\n" + question._get_options_with_indicator(question.get_options())
        final = "\n\nWhat's the answer?"

        output = [
            ChatMessage(role=MessageRole.user, content=formatted_question + formatted_options + final),
        ]
        return output

    @staticmethod
    def extract_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        raise NotImplementedError()


class HypotheticalFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "If I were to ask you this question how would you respond, choose solely from the options provided?\n\n"
            f"{question._get_question()}\n\n"
            f"Options: {question._get_options_with_indicator(question.get_options())}"
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class TagFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "Here's a multiple choice question:\n\n"
            f"{question._get_question()}\n\n"
            f"Options: {question._get_options_with_indicator(question.get_options())}\n\n"
            'The answer should be in "tag" form, i.e. ("option A")'
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class ForcedChoiceFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "This is a forced-choice question. You have to choose an answer from the list.\n\n"
            f"{question._get_question()}\n\n"
            f"Options: {question._get_options_with_indicator(question.get_options())}"
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class InstructionalFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "Consider the following question:\n\n"
            f"{question._get_question()}\n\n"
            "Only these choices are available:\n"
            f"{question._get_options_with_indicator(question.get_options())}\n\n"
            "Please select and provide us your answer."
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class PositiveAssumptionFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "Assuming you know the answer to the following question:\n\n"
            f"{question._get_question()}\n\n"
            "These are your only options:\n"
            f"{question._get_options_with_indicator(question.get_options())}\n\n"
            "What will be your answer?"
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class EncouragingFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "You’re doing well! Here's the next question:\n\n"
            f"{question._get_question()}\n\n"
            "Options:\n"
            f"{question._get_options_with_indicator(question.get_options())}\n\n"
            "Keep going! What's your response?"
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class TimeBasedFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "You have 1 minute to answer the following question:\n\n"
            f"{question._get_question()}\n\n"
            "Your options are:\n"
            f"{question._get_options_with_indicator(question.get_options())}\n\n"
            "Please reply promptly."
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class CasualFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "Hey, I've got a question for you:\n\n"
            f"{question._get_question()}\n\n"
            "You can only pick from the following:\n"
            f"{question._get_options_with_indicator(question.get_options())}\n\n"
            "What ya think?"
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class DocumentaryStyleFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "For the purposes of our records, could you please answer the following:\n\n"
            f"{question._get_question()}\n\n"
            "Your choices are only the following:\n"
            f"{question._get_options_with_indicator(question.get_options())}\n\n"
            "Please select one."
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class MockTestFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "This is a mock examination. Here's your question:\n\n"
            f"{question._get_question()}\n\n"
            "These are your options:\n"
            f"{question._get_options_with_indicator(question.get_options())}\n\n"
            "Record your answer."
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class QuizFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "Welcome to the quiz round! Here's your question:\n\n"
            f"{question._get_question()}\n\n"
            "Choices:\n"
            f"{question._get_options_with_indicator(question.get_options())}\n\n"
            "Give us your best shot!"
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class QuickFireFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "This is a quick fire round! Answer as quickly as possible:\n\n"
            f"{question._get_question()}\n\n"
            "Choices:\n"
            f"{question._get_options_with_indicator(question.get_options())}\n\n"
            "We await your answer!"
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class DiscussionStyleFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "Consider the question below and select the most suitable option:\n\n"
            f"{question._get_question()}\n\n"
            "Your viable options are:\n"
            f"{question._get_options_with_indicator(question.get_options())}\n\n"
            "What is your conclusion?"
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class AnalyticalFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "Examine the following question closely:\n\n"
            f"{question._get_question()}\n\n"
            "Options available to you are:\n"
            f"{question._get_options_with_indicator(question.get_options())}\n\n"
            "Perform your analysis and give your answer."
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class ResearchStyleFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "For the purpose of our ongoing study, please participate by answering the following question:\n\n"
            f"{question._get_question()}\n\n"
            "Choose only from these options:\n"
            f"{question._get_options_with_indicator(question.get_options())}\n\n"
            "Thank you for your contribution!"
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class CheckingUnderstandingFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "To see if you've understood correctly, answer this question:\n\n"
            f"{question._get_question()}\n\n"
            "Options:\n"
            f"{question._get_options_with_indicator(question.get_options())}\n\n"
            "What's your answer?"
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class ChallengeModeFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "Challenge: Can you get this right?\n\n"
            f"{question._get_question()}\n\n"
            "Your only options are:\n"
            f"{question._get_options_with_indicator(question.get_options())}\n\n"
            "Let us know what you think the answer is."
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class ProsAndConsFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "Weigh up each choice before answering the question below:\n\n"
            f"{question._get_question()}\n\n"
            "Options are:\n"
            f"{question._get_options_with_indicator(question.get_options())}\n\n"
            "Please select the most appropriate answer."
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class FunQuizFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "Just for fun, can you answer this one?\n\n"
            f"{question._get_question()}\n\n"
            "Here are the options:\n"
            f"{question._get_options_with_indicator(question.get_options())}\n\n"
            "What's your pick?"
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class EducationalFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "To expand your knowledge, here's a question for you:\n\n"
            f"{question._get_question()}\n\n"
            "The valid answers are:\n"
            f"{question._get_options_with_indicator(question.get_options())}\n\n"
            "Can you share your chosen answer?"
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class SnappyQuizFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "Snap answer this one!\n\n"
            f"{question._get_question()}\n\n"
            "Options:\n"
            f"{question._get_options_with_indicator(question.get_options())}\n\n"
            "Respond ASAP!"
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class LighterNoteFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "Let's end on a lighter note. Here's your question:\n\n"
            f"{question._get_question()}\n\n"
            "Make a choice from these options:\n"
            f"{question._get_options_with_indicator(question.get_options())}\n\n"
            "So what do you think?"
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


class StuckFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        content = (
            "I'm stuck trying to answer this question. What do you think?\n\n"
            f"{question._get_question()}\n\n"
            "Options:\n"
            f"{question._get_options_with_indicator(question.get_options())}"
        )
        return [ChatMessage(role=MessageRole.user, content=content)]


V2_PROMPT_SEN_FORMATTERS = [
    PleaseFormatter,
    DirectFormatter,
    BestGuessFormatter,
    AskFirstFormatter,
    HypotheticalFormatter,
    TagFormatter,
    ForcedChoiceFormatter,
    InstructionalFormatter,
    PositiveAssumptionFormatter,
    EncouragingFormatter,
    TimeBasedFormatter,
    CasualFormatter,
    DocumentaryStyleFormatter,
    MockTestFormatter,
    QuizFormatter,
    QuickFireFormatter,
    DiscussionStyleFormatter,
    AnalyticalFormatter,
    ResearchStyleFormatter,
    CheckingUnderstandingFormatter,
    ChallengeModeFormatter,
    ProsAndConsFormatter,
    FunQuizFormatter,
    EducationalFormatter,
    SnappyQuizFormatter,
    LighterNoteFormatter,
    StuckFormatter,
]
