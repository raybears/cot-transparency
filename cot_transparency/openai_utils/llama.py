from threading import Lock
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from cot_transparency.data_models.messages import StrictChatMessage, StrictMessageRole
from cot_transparency.data_models.config import OpenaiInferenceConfig


class Llama27BHelper:
    def __init__(self, pretrained_model: str = "meta-llama/Llama-2-7b-chat-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        token = os.getenv("HF_TOKEN", None)
        if token is None:
            raise ValueError("No HF_TOKEN environment variable found. Must be specifed in env")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_auth_token=token)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model, use_auth_token=token).to(self.device)

    def generate_text(
        self, prompt: str, max_length: int = 100, temperature: float = 1.0, top_p: Optional[float] = 1.0
    ) -> str:
        if temperature == 0.0:
            do_sample = False
        else:
            do_sample = True

        if top_p is None:
            top_p = 1.0

        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(
            inputs.input_ids.to(self.device),
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
        )
        output = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # strip the prompt from the response
        output = output.replace(prompt.lstrip("<s>"), "")
        return output


llama_cache = {}
lock = Lock()


def llama_v2_prompt(messages: list[StrictChatMessage]) -> str:
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""  # noqa

    if messages[0].role != StrictMessageRole.system:
        messages = [
            StrictChatMessage(role=StrictMessageRole.system, content=DEFAULT_SYSTEM_PROMPT),
        ] + messages

    assert messages[1].role == StrictMessageRole.user
    messages = [
        StrictChatMessage(
            role=StrictMessageRole.user, content=B_SYS + messages[0].content + E_SYS + messages[1].content
        )
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt.content).strip()} {E_INST} {(answer.content).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    if len(messages) % 2 == 1:
        messages_list.append(f"{BOS}{B_INST} {(messages[-1].content).strip()} {E_INST}")

    if messages[-1].role == StrictMessageRole.assistant:
        # strip the EOS as we want the model to continue
        messages_list[-1] = messages_list[-1].replace(f" {EOS}", "")

    return "".join(messages_list)


def call_llama_chat(prompt: list[StrictChatMessage], config: OpenaiInferenceConfig) -> str:
    supported_models = set("Llama-2-7b-chat-hf")

    assert config.model in supported_models, f"llama model {config.model} is not supported yet"
    formatted_prompt = llama_v2_prompt(prompt)

    with lock:
        if config.model in llama_cache:
            chat_model = llama_cache[config.model]
        else:
            chat_model = Llama27BHelper(config.model)
            llama_cache[f"meta-llama/{config.model}"] = chat_model

    return chat_model.generate_text(
        formatted_prompt, max_length=config.max_tokens, temperature=config.temperature, top_p=config.top_p
    )
