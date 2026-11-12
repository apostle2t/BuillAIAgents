
"""
Goal: I want to build a voice AI Agent.

We are using the pipeline architecture.

Things needed:
1. The transport medium (for example, a phone call, a web app, a mobile app, etc.)
2. Turn detection
3. Speech to text (STT)
4. LLM (Large Language Model)
5. Text to speech (TTS)

"""

from smolagents import CodeAgent
from fastrtc import get_stt_model, get_tts_model, Stream, ReplyOnPause, get_twilio_turn_credentials
from dotenv import load_dotenv
import os

load_dotenv()

# we build the Agent
model = OpenAIServerModel(
    model_id="deepseek/deepseek-v4.1-flash",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

agent_voice = CodeAgent(
    model=model,
    tools=[
    ],
    description="You are a helpful voice assistant. You can answer questions and provide information to the user",
)

# We define the SST and TTS models
def voice_response():
    pass