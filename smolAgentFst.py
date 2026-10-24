import os
from openai import OpenAI
from smolagents import CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool, load_tool, tool, \
    GradioUI, OpenAIServerModel
import datetime
import requests
import pytz
import yaml
from dotenv import load_dotenv

"""
We are building an agent which generates Cat Images, does websearch, Checks the local timezone in a 
particular place
"""

load_dotenv()


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
        timezone (str): A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
        return f"The current local time in {timezone} is: {local_time}"
    except pytz.exceptions.UnknownTimeZoneError:
        return f"Experienced an error fetching the timezone at: {timezone}"


# We define the Agent
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

final_answer = FinalAnswerTool()

# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True, token = HF_TOKEN)

# We make use of the search tool
search_web = DuckDuckGoSearchTool()

# We define the model.
model = OpenAIServerModel(
    model_id="deepseek/deepseek-v4.1-flash",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    max_tokens=3000,
)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

# We're creating our CodeAgent
agent = CodeAgent(
    model=model,
    tools=[final_answer, get_current_timezone, image_generation_tool, search_web],  # add your tools here (don't remove final_answer)
    max_steps=6,
    verbosity_level=1,
    instructions = "You are a helpful assistant that can generate images, search the web, and check the time anywhere.",
)

GradioUI(agent).launch()