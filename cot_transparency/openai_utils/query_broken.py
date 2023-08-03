from dotenv import load_dotenv
import openai

load_dotenv()

with open("broken_prompt.txt", "r") as fh:
    text = fh.read()


response = openai.Completion.create(
    model="text-davinci-002",
    prompt=text,
)

print(text)
print(response)
