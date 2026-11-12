
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
from typing import Generator
from smolagents import CodeAgent, OpenAIServerModel
from fastrtc import get_stt_model, get_tts_model, Stream, ReplyOnPause, get_twilio_turn_credentials
from dotenv import load_dotenv
import numpy as np
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

# We definbe the STT and TTS models
sst_model = get_stt_model()
tts_model = get_tts_model()

# We define the full system prompt for the LLM
SYSTEM_PROMPT = """

# PERSONALITY
# GOAL
# TONE
# GUARDRAILS



"""


# We define the SST and TTS models
def voice_response(audio_input: tuple[int, np.ndarray]) -> Generator[tuple[int, np.ndarray], None, None]:
    text_data = sst_model(audio_input)
    if text_data is None:
        yield None
    else:
        input_text = f"{SYSTEM_PROMPT}\n\nUser: {text_data}\n\nAssistant: "
        response = agent_voice(input_text)
    for audio_output in tts_model.stream_tts_sync(response):
        yield audio_output


# we stream the response back to the user
stream = Stream()