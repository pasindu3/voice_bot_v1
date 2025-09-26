import os, asyncio, tempfile
import streamlit as st
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
import edge_tts   # fast local TTS
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
st.set_page_config(page_title="Real-time VoiceBot", layout="centered")

st.title("🎤 Streaming Voice Bot")

# Hold chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat log
for m in st.session_state.messages:
    st.markdown(f"**{m['role'].title()}:** {m['content']}")

async def speak_stream(text_stream):
    """Convert arriving text chunks to speech as soon as they appear."""
    communicator = edge_tts.Communicate(text_stream, voice="en-US-JennyNeural")
    async for chunk in communicator.stream():
        if chunk["type"] == "audio":
            # Play audio bytes instantly in browser
            b64 = chunk["data"].encode("base64") if hasattr(chunk["data"], "encode") else chunk["data"]
            st.markdown(
                f"""<audio autoplay>
                       <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    </audio>""",
                unsafe_allow_html=True
            )

async def process_user_audio(path):
    # 1️⃣ STT
    with open(path, "rb") as f:
        transcript = await asyncio.to_thread(
            client.audio.transcriptions.create,
            model="whisper-1",
            file=f
        )
    user_text = transcript.text
    st.session_state.messages.append({"role":"user","content":user_text})
    st.chat_message("user").write(user_text)

    # 2️⃣ GPT streaming + live TTS
    bot_placeholder = st.chat_message("assistant")
    text_collector = []

    with client.chat.completions.stream(
        model="gpt-4o",
        messages=st.session_state.messages
    ) as stream:
        for event in stream:            # ✅ regular for-loop
            if event.type == "delta":
                chunk = event.delta.get("content", "")
                if chunk:
                    text_collector.append(chunk)
                    bot_placeholder.write("".join(text_collector))
                    # Speak chunk immediately
                    await speak_stream(chunk)

    full_reply = "".join(text_collector)
    st.session_state.messages.append({"role":"assistant","content":full_reply})

def main():
    st.info("Click mic, speak, and listen to the bot respond **while it thinks**.")
    audio = mic_recorder(start_prompt="🎤 Start", stop_prompt="⏹️ Stop", key="rec")

    if audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio["bytes"])
            temp_path = f.name
        asyncio.run(process_user_audio(temp_path))

if __name__ == "__main__":
    main()
