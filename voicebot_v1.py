import os, asyncio, tempfile
import streamlit as st
from openai import OpenAI
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import soundfile as sf
import os
from dotenv import load_dotenv

load_dotenv()



openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Fast Voice Bot", layout="centered")
st.title("🎙️ Real-Time Voice Bot")

status_placeholder = st.empty()

async def transcribe_and_respond(audio_path):
    # 1️⃣ Speech to Text (Whisper)
    with open(audio_path, "rb") as f:
        stt = await asyncio.to_thread(
            openai_client.audio.transcriptions.create,
            model="whisper-1",
            file=f
        )
    user_text = stt.text
    status_placeholder.markdown(f"**You:** {user_text}")

    # 2️⃣ GPT-4o response
    llm = await asyncio.to_thread(
        openai_client.chat.completions.create,
        model="gpt-4o",
        messages=[{"role":"user","content":user_text}]
    )
    bot_text = llm.choices[0].message.content
    status_placeholder.markdown(f"**Bot:** {bot_text}")

    # 3️⃣ Text to Speech (fast streaming TTS)
    tts_resp = await asyncio.to_thread(
        openai_client.audio.speech.create,
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=bot_text
    )
    out_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts_resp.stream_to_file(out_audio.name)
    return out_audio.name

def main():
    st.write("Press record, speak, and wait for a quick voice reply.")

    # Simple mic recorder widget
    from streamlit_mic_recorder import mic_recorder
    audio = mic_recorder(start_prompt="🎤 Start Recording",
                         stop_prompt="⏹️ Stop",
                         key="recorder")

    if audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio["bytes"])
            temp_audio_path = f.name

        # Run pipeline concurrently
        audio_out_path = asyncio.run(transcribe_and_respond(temp_audio_path))
        st.audio(audio_out_path, format="audio/mp3")

if __name__ == "__main__":
    main()
