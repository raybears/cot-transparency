import os
os.environ["TRANSFORMERS_CACHE"] = "/scratch/huggingface"

from transformers import AutoTokenizer
import transformers
import torch
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import uvicorn

class Query(BaseModel):
    prompt: str

app = FastAPI()

pipeline = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    global pipeline, tokenizer
    pipeline = transformers.pipeline(
        "text-generation",
        model=MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        use_auth_token="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",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# TODO are these methods really preferable to just accessing globals from run_inference?
def get_pipeline():
    return pipeline

def get_tokenizer():
    return tokenizer


@app.post("/llama")
async def run_inference(query: Query, pipeline = Depends(get_pipeline), tokenizer = Depends(get_tokenizer)):
    print("run query")
    sequences = pipeline(
        f"{query.prompt}\n",
        # do_sample=True,
        top_k=10,
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=500,
    )
    print("done")
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")


if __name__ == "__main__":
    MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
    uvicorn.run(app, host="0.0.0.0", port=8042)
