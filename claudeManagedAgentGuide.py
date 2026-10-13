
from dotenv import load_dotenv
from anthropic import Anthropic
import os

# We get the Anthropic API KEY from the .env file with the help of dotenv.
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_AGENT_ID = os.getenv("ANTHROPIC_AGENT_ID")
ANTHROPIC_ENVIRONMENT_ID = os.getenv("ANTHROPIC_ENVIRONMENT_ID")
ANTHROPIC_SESSION_ID = os.getenv("ANTHROPIC_SESSION_ID")


# # We create the Agent.
client = Anthropic()
# agent = client.beta.agents.create(
#     name="Research Assistant",
#     model="claude-haiku-5-5",
#     system="You are a helpful research assistant. Write clean, well-documentation.",
#     tools=[
#         {"type": "agent_toolset_20260401"},
#     ],
# )

#agent = client.beta.agents.update(
#     ANTHROPIC_AGENT_ID,
#     name="Research Assistant",
#     model="claude-haiku-5-5",
#     system="""You are a research assistant for Berlin public transport (VBB/BVG) data.
#     Use the API at https://v6.vbb.transport.rest. Read its docs first, then call it with
#     Python requests or curl from the sandbox.
#     - Resolve a stop name to a stop ID with the locations endpoint before asking for departures.
#     - Stay under 100 requests per minute and cache results in files.
#     - Report disruptions from the remarks data, with the line and time.
#     - Never invent departures or delays. If a call fails, report the exact error.
#     - Write clean, well-documented code with short docstrings.""",
#     tools=[
#         {"type": "agent_toolset_20260401"},
#     ],
# )


# We set up the environment in which the agent will operate.
# environment = client.beta.environments.create(
#     name="Research Environment",
#     config={
#         "type": "cloud",
#         "networking": {"type": "limited", "allow_package_managers": True},
#     },
# )

# environment = client.beta.environments.update(
#     ANTHROPIC_ENVIRONMENT_ID,
#     name="Research Environment",
#     config={
#         "type": "cloud",
#         "networking": {"type": "limited", "allow_package_managers": True, "allowed_hosts": ["v6.vbb.transport.rest"]},
#     },
# )


# We create a session for the agent to run in.
# session = client.beta.sessions.update(
#     agent=ANTHROPIC_AGENT_ID,
#     environment_id=ANTHROPIC_ENVIRONMENT_ID,
#     title="Quick Research session",
# )


with client.beta.sessions.events.stream(ANTHROPIC_SESSION_ID) as stream:
    # Send the user message after the stream opens
    client.beta.sessions.events.send(
        ANTHROPIC_SESSION_ID,
        events=[
            {
                "type": "user.message",
                "content": [
                    {
                        "type": "text",
                        "text": "Next 5 departures from Alexanderplatz, and any disruptions on the U2?",
                    },
                ],
            },
        ],
    )
    # Process streaming events
    for event in stream:
        match event.type:
            case "agent.message":
                for block in event.content:
                    if block.type == "text":
                        print(block.text, end="")
            case "agent.tool_use":
                print(f"\n[Using tool: {event.name}]")
            case "session.status_idle":
                print("\n\nAgent finished.")
                break




