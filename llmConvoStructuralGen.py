"""
This python file shows how an LLM Structure their generations in a conversational context.
"""
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from openai import OpenAI
import os

# We get the token from the .env file with the help of dotenv.
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# We define the chat client with the InferenceClient class from the huggingface_hub library. We pass the token to the constructor of the InferenceClient class. This will allow us to use the Hugging Face API to generate text.
client = OpenAI(
    base_url = "https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY)


SYSTEM_PROMPT = """

# Personality
You are a helpful personal assistant.

# Tool
get_my_name: This returns the name of the Engineer managing this code. Just call the tool and it will return the name of the Engineer managing this code. The tool takes no input and returns a string as output.
This is how to use the tool:
Anytime this tool is called, I want you to specify a json blob using the example below in this format:
{
    "tool_name": "get_my_name"
}

# Goal
Answer the user's question as best you can. Use the tool only when you need information you don't already have.

# Format
ALWAYS follow this format:

Question: the question you must answer
Thought: think about what to do next. Only one action at a time.
Action: the json blob for the tool you want to call
Observation: the result of the tool. This will be given to you, so do not write it yourself.

... (Thought/Action/Observation can repeat as many times as needed)

When you have the answer, end with:
Thought: I now know the final answer
Final Answer: the final answer to the original question

# Rules
- After writing an Action, STOP and wait for the Observation.
- Always use the exact characters `Final Answer:` when giving your final answer.

"""


# # We now define the message template
message = [
{   "role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is the name of the Engineer managing this code?"}
]

response = client.chat.completions.create(
    model="deepseek/deepseek-v4.1-flash",
    messages=message,
    max_tokens=500,
    stream = False,
    stop = ["Observation:"],
    extra_body = {"reasoning": {"enabled": False}}
)
print(response.choices[0].message.content)


# This is a simple function that returns my name. It is meant to be a tool that an LLM can call to get my name. The function takes a string as input and returns a string as output.
def get_my_name() -> str:
    """
    This function returns my name.
    """
    name : str = "Kimi"
    return f"My name is {name}."
get_my_name()


# We can now call the client with the system prompt and the message template. We will pass the system prompt to the chat method. The chat method will return a response from the model. We will print the response to the console.
message = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is the name of the Engineer managing this code?"}
]

message = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is the name of the Engineer managing this code?"},
    {"role": "assistant", "content": response.choices[0].message.content + "\nObservation: " + get_my_name()}
]

response = client.chat.completions.create(
    model="deepseek/deepseek-v4.1-flash",
    messages=message,
    max_tokens=500,
    stream = False,
    extra_body = {"reasoning": {"enabled": False}}
)
print(response.choices[0].message.content)



