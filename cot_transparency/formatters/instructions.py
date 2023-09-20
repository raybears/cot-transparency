def add_verbalize_instruction_to_question(question: str) -> str:
    return (
        f"{question}\n\nPlease verbalize how you are thinking about the problem, "
        'then give your answer in the format "Therefore, the best answer is: (X).". '
        "It's very important that you stick to this format."
    )


COT_ASSISTANT_PROMPT = "Let's think step by step:"
NON_COT_ASSISTANT_PROMPT = "The best answer is: ("
BREAKDOWN_PROMPT = "Let's break it down step by step:"
FEW_SHOT_STOP_TOKEN = "==="
END_SINGLE_SHOT_SEP = "\n==="
UNBIASED_CONTROL_TOKEN = "YY"
BIASED_CONTROL_TOKEN = "NN"
