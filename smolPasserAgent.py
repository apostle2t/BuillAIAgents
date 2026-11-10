

"""
We want to build an output passer Agent.

Steps Involved:
1. We define the tools required for this task.
For the tools, I am thinking we define like 8 to 10 fields for output.
output
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
def output_passer(headline: str, problem: str, extent: str, danger: str, address: str, language: str, summary: str, caller_name: str, caller_phone: str) -> json:
    """
    This tool takes in various fields related to a report and returns a JSON object containing the information.
    Args:
        headline (str): Short German headline of the problem, at most 8 words, no names and no address, for example: Heizung in der ganzen Wohnung ausgefallen.

        problem (str): Exactly one of: heating (heating, hot water), water (burst pipe, water damage, blocked drain), electricity (power cut, fuse), gas (gas smell, gas appliance), lift, lock_door (locked out, lock, front door), tv_cable (TV, cable, satellite), noise (noise, neighbours), damage (other damage), general_question (questions, contact data, house rules), other.

        extent (str): Exactly one of: room (one room), flat (whole flat), building (whole building), unknown (the caller does not know). null if not discussed.

        danger (str): "yes if people may be in danger (gas or fire smell, smoke, water near electrical equipment, injured or trapped people) or the caller is afraid; no if the caller said nobody is in danger; unknown if it stayed unclear; null if not discussed."

        address (str): "Address of the problem: street as the caller said it, house number always as digits (5, not 'fünf' or 'five'), then floor and flat in German with digits (for example 'Musterweg 5, 3. Stock links'), even if the call was in another language."

        language (str): "Language the caller spoke, as a two-letter code, e.g. de, en, pl, tr."

        summary (str): Two factual sentences in German: what is wrong and what the caller needs. No personal names.

        caller_name (str): "The caller's name exactly as they said it (first and/or last name). null if no name was said."

        caller_phone (str): "Only if the caller gave a different phone number for the SMS or the call back than the one they are calling from: that number in international format, e.g. +4917012345678. Otherwise null."
    Returns:
        json: A JSON object containing the information.
    """
    return json.dumps({
        "headline": headline,
        "problem": problem,
        "extent": extent,
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
    instructions = """
    You read the transcript of a call to an emergency hotline (Notdienst) and fill in the case fields.
        Rules:
        - Only use facts that were said in the call. Do not invent or guess anything.
        - Everything must be outputed in English.  
        - If something was not said, return null for that field.
        - Pick category values only from the allowed values given for each field. If a category is unclear, use other.
        - Write the title and the summary in English. Keep addresses exactly as the caller said them.
        - Put the caller's name only into caller_name. Do not include it or any other personal names in title, summary or address.
    """
                    )

response = passer_agent.run(
"Agent: Guten Tag, hier ist der Notdienst der Musterwohnbau. Sie sprechen mit einem KI-Assistenten, Ihre Angaben werden zur Bearbeitung schriftlich festgehalten. Was kann ich für Sie tun?\nCaller: Guten Abend. In meiner Küche riecht es etwas seltsam, vielleicht nach Gas. Ich bin mir nicht sicher.\nAgent: Bitte verlassen Sie die Wohnung, schalten Sie nichts ein und rufen Sie eins eins zwei an. Ich verbinde Sie jetzt mit einem Kollegen.",)

print("Response from the output passer Agent:")
print(response)





