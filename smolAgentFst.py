import os
from openai import OpenAI
from smolagents import CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool, InferenceClientModel, load_tool, tool, GradioUI
import datetime
import requests
import pytz
import yaml

"""
We are building an agent which generates Cat Images, does websearch, Checks the local timezone in a 
particular place

"""


# We define the tool blocks
@tool
def my_default_too(arg1: str, arg2: str) -> str:
    """
    This is my default tool. It takes two arguments and returns a string.
    Args:
        arg1 (str): The first argument.
        arg2 (str): The second argument.
    """
    return f"My default tool was called with arguments: {arg1} and {arg2}."


@tool
def get_current_timezone(timezone: str) -> str:
    """
    A tool that fetched the current local time in a specified timezone.
    Args:
        timezone (str): A string representing a valid time (e.g., 'America/New_York').
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
        return f"The current local time in {timezone} is: {local_time}"
    except pytz.exceptions.UnknownTimeZoneError:
        return f"Experienced an error fetching the timezone at: {timezone}"


# We define the Agent
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
final_answer = FinalAnswerTool()

# We define the chat client with the InferenceClient class from the huggingface_hub library. We pass the token to the constructor of the InferenceClient class. This will allow us to use the Hugging Face API to generate text.
client = OpenAI(
    base_url = "https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY)

model = client.chat.completions.create(
    model="deepseek/deepseek-v4.1-flash",
    max_tokens=3000,
    stream = False,
    extra_body = {"reasoning": {"enabled": False}}
)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

# We're creating our CodeAgent
agent = CodeAgent(
    model=model,
    tools=[final_answer, get_current_timezone],  # add your tools here (don't remove final_answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

GradioUI(agent).launch()