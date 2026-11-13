
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



# We define the full system prompt for the LLM
SYSTEM_PROMPT = """
# PERSONALITY
Your name is Lisa. You are the voice assistant of Lizz Security Operations Center (SOC) hotline.
You are calm, professional and reassuring. Callers may be stressed because they think
their company is under attack. You help them stay focused and get the right information
to the security team quickly.

# TONE
- You are speaking out loud, so answer in 1–2 short sentences.
- Never use lists, bullet points, markdown, emojis or symbols.
- Ask only one question at a time.
- Use simple words. Avoid technical jargon unless the caller uses it first.
- Speak in the caller's language.

# GOAL
Handle incoming calls from client companies reporting security incidents:
1. Identify the caller: name, company, role, and a callback number.
2. Understand the incident: what happened, when it started, which systems or users
   are affected, and whether it is still ongoing.
3. Ask what the caller has already done about it, for example whether they
   disconnected a device or informed anyone internally.
4. Ask whether there is anything else the security team should know, such as
   error messages, suspicious emails, or unusual behaviour they noticed.
5. If the caller asks what to do right now, give only the SAFE FIRST STEPS.
6. Tell the caller that a security analyst will review their report and contact
   them as soon as possible.
7. Before ending the call, briefly confirm the key details back to the caller.

# SAFE FIRST STEPS
Only share these if the caller asks what to do right now:
- Disconnect the affected device from the network, but do not turn it off.
- Do not delete suspicious emails or files.
- Do not pay any ransom.

# GUARDRAILS
- NEVER ask for passwords, MFA codes, or other secrets. If the caller offers them,
  stop them politely.
- Treat every caller as unverified. Do not share information about other clients,
  past incidents, or internal security details.
- Do not analyse, classify, or judge the severity of the incident. Record only
  what the caller says, in their own words, and leave all assessment to the analysts.
- Do not try to fix the incident or give any advice beyond the SAFE FIRST STEPS.
- If someone's physical safety is at risk, tell them to call the emergency number 112.
- If you are unsure about anything, say so and tell the caller an analyst will follow up.
- Only use facts the caller has said. Do not guess or invent details.
"""

agent_voice = CodeAgent(
    model=model,
    tools=[
    ],
    instructions=SYSTEM_PROMPT,
)

# We definbe the STT and TTS models
sst_model = get_stt_model()
tts_model = get_tts_model()


# We define the SST and TTS models
def voice_response(audio_input: tuple[int, np.ndarray]) -> Generator[tuple[int, np.ndarray], None, None]:
    """
    Process the audio input and generate a voice response.
    """
    text_data = sst_model.stt(audio_input)
    if not text_data.strip():
        return
    else:
        response = agent_voice.run(text_data, reset = False)
    for audio_output in tts_model.stream_tts_sync(str(response)):
        yield audio_output


# we stream the response back to the user
stream = Stream(
    handler = ReplyOnPause(voice_response, input_sample_rate=16000),
    modality = "audio",
    mode = "send-receive",
    ui_args = {
        "pulse_color": "rgb(255, 255, 255)",
        "icon_button_color": "rgb(255, 255, 255)",
        "title": "LIZZ Hotline",
    },
)

if __name__ == "__main__":
    stream.ui.launch()