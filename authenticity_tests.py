
def split_text_preserve_case(text, separator):
    parts = None

    if separator.lower() in text:
        parts = text.split(separator.lower(), maxsplit=1)
        parts[1] = separator.lower() + parts[1]
    elif separator in text:
        parts = text.split(separator, maxsplit=1)
        parts[1] = separator + parts[1]
    else:
        raise ValueError(f"Separator '{separator}' not found in text")
    return parts
    
def get_blank_cot_inp(inp, reasoning, tokenizer):
    cot_part, ans_part = split_text_preserve_case(reasoning, "The best answer")

    num_reasoning_tokens = len(tokenizer.encode(cot_part))
    blank_cot = ' '.join(['...' for _ in range(num_reasoning_tokens)])
    return f"{inp}\n\n" + blank_cot + " " + ans_part

