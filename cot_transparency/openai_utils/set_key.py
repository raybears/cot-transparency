import os

import openai
from dotenv import load_dotenv


def set_keys_from_env():
    # take environment variables from .env so you don't have
    # to source .env in your shell
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key
