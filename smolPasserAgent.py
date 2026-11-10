

"""
We want to build an output passer Agent.

Steps Involved:
1. We define the tools required for this task.
For the tools, I am thinking we define like 8 to 10 fields.
- The headline of the report
- The problem mentioned in the report
- The severity of the report
- Danger: If any danger is mentioned in the report
- The address mentioned in the report
- language: The language of the report
- Summary: A summary of the report
- Caller's name: The name of the caller
- Caller's phone number: The phone number of the caller

2. We define the brain (model that the Agent will use).


3. We define the Agent (Setting up the Agent with the brain and tools).
"""

from smolagents import CodeAgent, load_tool, tool, OpenAIServerModel
import json, os
from dotenv import load_dotenv

load_dotenv()

#1
@tool
def output_passer(headline: str, problem: str, severity: str, danger: str, address: str, language: str, summary: str, caller_name: str, caller_phone: str) -> json:
    """
    This tool takes in various fields related to a report and returns a JSON object containing the information.
    Args:
        headline (str): The headline of the report.
        problem (str): The problem mentioned in the report.
        severity (str): The severity of the report.
        danger (str): If any danger is mentioned in the report.
        address (str): The address mentioned in the report.
        language (str): The language of the report.
        summary (str): A summary of the report.
        caller_name (str): The name of the caller.
        caller_phone (str): The phone number of the caller.
    Returns:
        json: A JSON object containing the information.
    """
    return json.dumps({
        "headline": headline,
        "problem": problem,
        "severity": severity,
        "danger": danger,
        "address": address,
        "language": language,
        "summary": summary,
        "caller_name": caller_name,
        "caller_phone": caller_phone
    }, indent=4)

#2
model = OpenAIServerModel(
    model_id="deepseek/deepseek-v4.1-flash",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    max_tokens=3000,
)

#3
passer_agent = CodeAgent(
    model = model,
    tools = [output_passer],
    max_steps = 3,
    instructions = "You are an output passer Agent. You will be given a report with various fields. Your task is to pass the information to the output_passer tool and return the JSON object."
)

response = passer_agent.run(
                            "The headline of the report is 'Fire at the warehouse'. The problem is that there is a fire at the warehouse. The severity is high. There is no danger mentioned. The address is 123 Main Street. The language of the report is English. The summary is that there is a fire at the warehouse and it is a high severity incident.")

print("Response from the output passer Agent:")
print(response)





