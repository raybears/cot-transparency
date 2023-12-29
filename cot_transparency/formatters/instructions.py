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


# Note: Let's think step by step: gets removed in training time in
# task_output_to_finetune_sample. This is so that we don't overfit to the
# single Let's think step by step: prompt.
COT_ASSISTANT_PROMPT_TESTING = "Let's think step by step:"
NON_COT_ASSISTANT_PROMPT = "The best answer is: ("
BREAKDOWN_PROMPT = "Let's break it down step by step:"
FEW_SHOT_STOP_TOKEN = "==="
END_SINGLE_SHOT_SEP = "\n==="
UNBIASED_CONTROL_TOKEN = "YY"
BIASED_CONTROL_TOKEN = "NN"
