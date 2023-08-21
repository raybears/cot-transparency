import os
os.environ["TRANSFORMERS_CACHE"] = "/scratch/huggingface"

from transformers import AutoTokenizer
import transformers
import torch
import fire
from pathlib import Path
import dotenv

dotenv.load_dotenv()
print(os.environ["TRANSFORMERS_CACHE"])
print(os.environ["HF_TOKEN"])


PROMPT = """Correctly capitalise and punctuate this sentence without adding quotation marks: {}"""


def run_inference(model="meta-llama/Llama-2-7b-chat-hf", dbl="/exp/chrisw/np/test_data_20230731.dbl"):

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
        use_auth_token=os.environ["HF_TOKEN"],
    )
    tokenizer = AutoTokenizer.from_pretrained(model)

    torch.manual_seed(0)

    text = [line for path in Path(dbl).read_text().splitlines()
                 for line in Path(path).read_text().splitlines()][:5]
    
    for example in text:
        sequences = pipeline(
            PROMPT.format(example),
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=1024,
        )
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")


if __name__ == "__main__":
    fire.Fire(run_inference)
