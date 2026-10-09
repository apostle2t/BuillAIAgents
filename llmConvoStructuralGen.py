"""
This python file shows how an LLM Structure their generations in a conversational context.
"""
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# We get the token from the .env file with the help of dotenv.
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


# We define the chat client with the InferenceClient class from the huggingface_hub library. We pass the token to the constructor of the InferenceClient class. This will allow us to use the Hugging Face API to generate text.
client = InferenceClient(model = "moonshotai/Kimi-K2.5", token=HF_TOKEN)


# We now define the message template
message = [
    {"role": "user", "content": "What is the name of the Engineer managing this code?"}
]


# We can now call the client. We will be using the chat method of the InferenceClient class. We will pass the message template to the chat method. The chat method will return a response from the model. We will print the response to the console.
response = client.chat.completions.create(
    messages=message,
    max_tokens=100,
    stream = False,
    extra_body = {"thinking": {"type": "disabled"}}
)


SYSTEM_PROMPT = """







"""










# This is a simple function that returns my name. It is meant to be a tool that an LLM can call to get my name. The function takes a string as input and returns a string as output.
def get_my_name(name: str) -> str:
    """
    This function returns my name.
    """
    return f"My name is {name}."

