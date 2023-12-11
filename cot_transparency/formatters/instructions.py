VERBALIZE_INSTRUCTION = (
    "\n\nPlease think step by step and "
    'then give your answer in the format "Therefore, the best answer is: (X).". '
    "It's very important that you stick to this format."
)


def add_verbalize_instruction_to_question(question: str) -> str:
    return f"{question}{VERBALIZE_INSTRUCTION}"


VERBALIZE_INSTRUCTION_NO_MCQ = "\n\nPlease think step by step and then give your best answer."


def add_verbalize_instruction_to_question_free_form(question: str) -> str:
    return f"{question}{VERBALIZE_INSTRUCTION_NO_MCQ}"


COT_ASSISTANT_PROMPT = ""  # Horrible hack to remove this
NOT_BLANK_COT_ASSISTANT_PROMPT = "Let's think step by step:"
NON_COT_ASSISTANT_PROMPT = "The best answer is: ("
BREAKDOWN_PROMPT = "Let's break it down step by step:"
FEW_SHOT_STOP_TOKEN = "==="
END_SINGLE_SHOT_SEP = "\n==="
UNBIASED_CONTROL_TOKEN = "YY"
BIASED_CONTROL_TOKEN = "NN"
