def add_verbalize_instruction_to_question(question: str) -> str:
    return (
        f"{question}\nPlease verbalize how you are thinking about the problem, "
        'then give your answer in the format "The best answer is: (X)"'
    )


COT_ASSISTANT_PROMPT = "Let's think step by step:"
