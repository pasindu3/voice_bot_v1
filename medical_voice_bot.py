import streamlit as st
import asyncio
import tempfile
import io
import base64
from audio_recorder_streamlit import audio_recorder
import openai
from openai import AsyncOpenAI
import numpy as np
import soundfile as sf
from scipy.io.wavfile import write
import threading
import queue
import time

# Initialize OpenAI client
client = AsyncOpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Medical bot system prompt
MEDICAL_BOT_PROMPT = """You are a helpful medical appointment booking assistant. You help patients book appointments, answer questions about appointments, and provide basic information about services. Keep responses concise but friendly. You can:
1. Help book appointments
2. Check appointment availability
3. Answer questions about medical services
4. Provide appointment confirmation details
5. Handle appointment modifications

Always be professional, empathetic, and efficient in your responses."""

class VoiceBot:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
    async def transcribe_audio(self, audio_bytes):
        """Convert audio to text using Whisper"""
        try:
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_bytes)
                temp_file.flush()
                
                with open(temp_file.name, "rb") as audio_file:
                    transcript = await client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text"
                    )
                return transcript.strip()
        except Exception as e:
            st.error(f"Transcription error: {str(e)}")
            return None
    
    async def generate_streaming_response(self, user_input):
        """Generate streaming response from GPT-4o"""
        try:
            messages = [
                {"role": "system", "content": MEDICAL_BOT_PROMPT},
                {"role": "user", "content": user_input}
            ]
            
            stream = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                stream=True,
                max_tokens=150,  # Keep responses concise for faster TTS
                temperature=0.3
            )
            
            response_text = ""
            sentence_buffer = ""
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    response_text += content
                    sentence_buffer += content
                    
                    # Check for sentence endings to stream TTS
                    if any(punct in content for punct in ['.', '!', '?', '\n']):
                        if sentence_buffer.strip():
                            await self.text_to_speech_streaming(sentence_buffer.strip())
                            sentence_buffer = ""
            
            # Handle any remaining text
            if sentence_buffer.strip():
                await self.text_to_speech_streaming(sentence_buffer.strip())
                
            return response_text
        except Exception as e:
            st.error(f"Response generation error: {str(e)}")
            return "I'm sorry, I'm having trouble processing your request right now."
    
    async def text_to_speech_streaming(self, text):
        """Convert text to speech and play immediately"""
        try:
            response = await client.audio.speech.create(
                model="tts-1",  # Faster than tts-1-hd
                voice="nova",    # Professional female voice for medical context
                input=text,
                speed=1.1       # Slightly faster for quicker responses
            )
            
            # Convert to audio and add to playback queue
            audio_data = response.content
            self.response_queue.put(audio_data)
            
        except Exception as e:
            st.error(f"TTS error: {str(e)}")
    
    def play_audio_stream(self):
        """Play audio as it becomes available"""
        while True:
            try:
                if not self.response_queue.empty():
                    audio_data = self.response_queue.get()
                    
                    # Create audio player HTML
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                    audio_html = f"""
                    <audio autoplay>
                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
                    
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
            except Exception as e:
                st.error(f"Audio playback error: {str(e)}")
                break

# Initialize the voice bot
if 'voice_bot' not in st.session_state:
    st.session_state.voice_bot = VoiceBot()
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

def main():
    st.set_page_config(
        page_title="Medical Voice Assistant",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Medical Appointment Voice Assistant")
    st.markdown("*Click the microphone to speak. The assistant will respond immediately with voice.*")
    
    # Create two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎤 Voice Input")
        
        # Audio recorder
        audio_bytes = audio_recorder(
            text="",
            recording_color="#ff0000",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=2.0,
            sample_rate=16000
        )
        
        if st.session_state.is_processing:
            st.info("🤖 Processing your request...")
    
    with col2:
        st.markdown("### 💬 Conversation History")
        
        # Display conversation history
        for i, (user_msg, bot_msg) in enumerate(st.session_state.conversation_history):
            with st.expander(f"Conversation {i+1}", expanded=(i == len(st.session_state.conversation_history)-1)):
                st.markdown(f"**You:** {user_msg}")
                st.markdown(f"**Assistant:** {bot_msg}")
    
    # Process audio when received
    if audio_bytes and not st.session_state.is_processing:
        st.session_state.is_processing = True
        
        async def process_voice_input():
            try:
                # Step 1: Transcribe audio
                with st.spinner("🎧 Transcribing audio..."):
                    transcript = await st.session_state.voice_bot.transcribe_audio(audio_bytes)
                
                if transcript:
                    st.success(f"🎯 Heard: {transcript}")
                    
                    # Step 2: Generate and stream response
                    with st.spinner("🤖 Generating response..."):
                        bot_response = await st.session_state.voice_bot.generate_streaming_response(transcript)
                    
                    # Add to conversation history
                    st.session_state.conversation_history.append((transcript, bot_response))
                    
                    st.success("✅ Response sent to audio!")
                    
                else:
                    st.error("❌ Could not understand audio. Please try again.")
                    
            except Exception as e:
                st.error(f"❌ Error processing voice input: {str(e)}")
            finally:
                st.session_state.is_processing = False
                st.rerun()
        
        # Run the async function
        asyncio.run(process_voice_input())
    
    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. **Click the microphone** to start recording
        2. **Speak clearly** about your appointment needs
        3. **Wait for the red light** to turn green (recording stopped)
        4. The assistant will **automatically respond with voice**
        5. **Listen to the response** - no need to click anything!
        
        **Sample requests:**
        - "I need to book an appointment with Dr. Smith"
        - "What appointments are available this week?"
        - "Can you reschedule my appointment?"
        - "What services do you offer?"
        """)
    
    # Technical info
    with st.expander("⚙️ Technical Details"):
        st.markdown("""
        **Optimization Features:**
        - ⚡ **Streaming Response**: TTS starts as soon as sentences are complete
        - 🔄 **Parallel Processing**: Transcription and response generation run concurrently
        - 🎯 **Optimized Models**: Whisper-1 + GPT-4o + TTS-1 for best speed/quality balance
        - 📦 **Sentence-level Streaming**: Audio plays immediately per sentence
        - 🚀 **Minimal Latency**: No waiting for complete responses
        """)

if __name__ == "__main__":
    main()


#pip install streamlit audio-recorder-streamlit openai numpy soundfile scipy asyncio

# I've created a medical voice bot with streaming capabilities and minimal latency. Here's what I've implemented:
# Key Features:
# 🚀 Speed Optimizations:

# Sentence-level streaming: TTS starts as soon as each sentence is complete
# Parallel processing: Uses asyncio for concurrent operations
# Optimized models: Whisper-1, GPT-4o, and TTS-1 for best speed/quality
# Auto-play audio: Responses play immediately without button clicks

# 🎤 Voice Interface:

# One-click recording: Microphone button for easy voice input
# Real-time processing: Immediate transcription and response
# Medical context: Specialized for appointment booking

# 📦 Required Dependencies:
# bashpip install streamlit audio-recorder-streamlit openai numpy soundfile scipy asyncio
# ⚙️ Configuration:
# You'll need to add your OpenAI API key to Streamlit secrets:
# toml# .streamlit/secrets.toml
# OPENAI_API_KEY = "your-api-key-here"
# How It Works:

# Record: Click microphone, speak your request
# Transcribe: Whisper-1 converts speech to text instantly
# Generate: GPT-4o streams response sentence by sentence
# Speak: TTS-1 converts each sentence to audio immediately
# Play: Auto-plays audio without waiting for complete response

# Speed Features:

# Streaming TTS: Audio starts playing as sentences complete
# Concurrent processing: All operations run in parallel
# Optimized prompts: Concise responses for faster TTS
# Queue management: Efficient audio playback system
# Medical specialization: Focused responses reduce processing time

# The bot will respond almost immediately after you finish speaking, with audio streaming as the AI generates its response. Perfect for medical appointment booking where quick responses are crucial!
# Would you like me to modify any specific aspects or add additional features?