import streamlit as st
import streamlit_webrtc as webrtc
import openai
from supabase import create_client, Client
import io
import base64
import tempfile
import threading
import queue
import time
from datetime import datetime, timedelta
import re
from config import *
import logging
import os
import av
import numpy as np
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Session state initialization
if 'conversation_state' not in st.session_state:
    st.session_state.conversation_state = ConversationState.GREETING
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'audio_queue' not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
if 'current_audio' not in st.session_state:
    st.session_state.current_audio = None

class AudioProcessor:
    """Handle audio processing for voice input"""
    
    def __init__(self):
        self.audio_frames = []
        self.is_recording = False
    
    def add_frame(self, frame):
        """Add audio frame to buffer"""
        if self.is_recording:
            self.audio_frames.append(frame)
    
    def start_recording(self):
        """Start recording audio"""
        self.audio_frames = []
        self.is_recording = True
    
    def stop_recording(self):
        """Stop recording and return audio data"""
        self.is_recording = False
        if not self.audio_frames:
            return None
        
        # Convert frames to audio data
        audio_data = self._frames_to_audio_data(self.audio_frames)
        self.audio_frames = []
        return audio_data
    
    def _frames_to_audio_data(self, frames):
        """Convert audio frames to bytes"""
        try:
            # Combine all frames into a single audio buffer
            audio_buffer = b''.join([frame.to_ndarray().tobytes() for frame in frames])
            return audio_buffer
        except Exception as e:
            logger.error(f"Error converting frames to audio: {e}")
            return None

class VoiceBot:
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
    def speech_to_text(self, audio_data):
        """Convert speech to text using Whisper"""
        try:
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Transcribe using Whisper
            with open(temp_file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return transcript.text
        except Exception as e:
            logger.error(f"Speech-to-text error: {e}")
            return None
    
    def text_to_speech(self, text):
        """Convert text to speech using GPT-4-mini-TTS"""
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
            return response.content
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
            return None
    
    def get_gpt_response(self, user_input, conversation_history):
        """Get response from GPT-4 based on conversation state"""
        try:
            system_prompt = self._get_system_prompt()
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history
            for entry in conversation_history[-10:]:  # Keep last 10 exchanges
                messages.append({"role": "user", "content": entry.get("user", "")})
                messages.append({"role": "assistant", "content": entry.get("bot", "")})
            
            # Add current user input
            messages.append({"role": "user", "content": user_input})
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"GPT response error: {e}")
            return "I'm sorry, I'm having trouble processing your request. Please try again."
    
    def _get_system_prompt(self):
        """Get system prompt based on current conversation state"""
        base_prompt = """You are a helpful medical appointment booking assistant. You can answer general medical questions and help book appointments."""
        
        if st.session_state.conversation_state == ConversationState.GREETING:
            return base_prompt + " Start by greeting the user and asking if they want to book an appointment or have questions."
        
        elif st.session_state.conversation_state == ConversationState.VERIFICATION:
            return base_prompt + " You are in verification mode. Ask for the user's date of birth, first 3 letters of first name, and first 3 letters of last name."
        
        elif st.session_state.conversation_state == ConversationState.BOOKING:
            return base_prompt + " You are booking an appointment. Confirm the details and proceed with booking."
        
        return base_prompt

class DatabaseManager:
    def __init__(self):
        self.supabase = supabase
    
    def verify_patient(self, dob, first_name_part, last_name_part):
        """Verify patient using partial name matching"""
        try:
            # Query patients table
            response = self.supabase.table('patients').select('*').execute()
            
            if not response.data:
                return None
            
            # Find matching patient
            for patient in response.data:
                patient_dob = patient.get('dob', '')
                patient_first = patient.get('first_name', '').lower()
                patient_last = patient.get('last_name', '').lower()
                
                # Check date of birth
                if str(patient_dob) == str(dob):
                    # Check first 3 letters of names
                    if (patient_first[:3] == first_name_part.lower()[:3] and 
                        patient_last[:3] == last_name_part.lower()[:3]):
                        return patient
            
            return None
        except Exception as e:
            logger.error(f"Patient verification error: {e}")
            return None
    
    def get_available_doctors(self):
        """Get list of available doctors"""
        try:
            # This would typically query a doctors table
            # For now, return mock data
            return [
                {"id": 1, "name": "Dr. Smith", "specialty": "General Medicine"},
                {"id": 2, "name": "Dr. Johnson", "specialty": "Cardiology"},
                {"id": 3, "name": "Dr. Williams", "specialty": "Pediatrics"}
            ]
        except Exception as e:
            logger.error(f"Get doctors error: {e}")
            return []
    
    def book_appointment(self, patient_id, doctor_id, appointment_time):
        """Book an appointment"""
        try:
            appointment_data = {
                "patient_id": patient_id,
                "doctor_id": doctor_id,
                "appointment_time": appointment_time,
                "status": "scheduled"
            }
            
            response = self.supabase.table('appointments').insert(appointment_data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Booking error: {e}")
            return None

def extract_patient_info(text):
    """Extract patient information from user input"""
    # Simple regex patterns for extracting information
    dob_pattern = r'\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})\b'
    name_pattern = r'\b([A-Za-z]{3,})\b'
    
    dob_match = re.search(dob_pattern, text)
    names = re.findall(name_pattern, text)
    
    dob = dob_match.group(1) if dob_match else None
    first_name = names[0] if len(names) > 0 else None
    last_name = names[1] if len(names) > 1 else None
    
    return dob, first_name, last_name

def process_voice_input(audio_data):
    """Process voice input and return bot response"""
    voice_bot = VoiceBot()
    db_manager = DatabaseManager()
    
    # Convert speech to text
    user_text = voice_bot.speech_to_text(audio_data)
    if not user_text:
        return "I couldn't understand what you said. Please try again."
    
    # Add to conversation history
    st.session_state.conversation_history.append({"user": user_text, "bot": ""})
    
    # Process based on conversation state
    if st.session_state.conversation_state == ConversationState.GREETING:
        if "book" in user_text.lower() or "appointment" in user_text.lower():
            st.session_state.conversation_state = ConversationState.VERIFICATION
            bot_response = BOT_MESSAGES["verification_start"]
        else:
            bot_response = voice_bot.get_gpt_response(user_text, st.session_state.conversation_history)
    
    elif st.session_state.conversation_state == ConversationState.VERIFICATION:
        dob, first_name, last_name = extract_patient_info(user_text)
        
        if dob and first_name and last_name:
            # Verify patient
            patient = db_manager.verify_patient(dob, first_name, last_name)
            if patient:
                st.session_state.patient_data = patient
                st.session_state.conversation_state = ConversationState.BOOKING
                bot_response = f"Hello {patient['first_name']}! I found your record. Let me book your appointment."
            else:
                st.session_state.conversation_state = ConversationState.FALLBACK
                bot_response = BOT_MESSAGES["verification_failed"]
        else:
            bot_response = "I need your date of birth, first 3 letters of your first name, and first 3 letters of your last name."
    
    elif st.session_state.conversation_state == ConversationState.FALLBACK:
        # Handle spelling of names
        bot_response = voice_bot.get_gpt_response(user_text, st.session_state.conversation_history)
        # Could implement more sophisticated name matching here
    
    elif st.session_state.conversation_state == ConversationState.BOOKING:
        # Book appointment
        doctors = db_manager.get_available_doctors()
        if doctors:
            doctor = doctors[0]  # Take first available doctor
            appointment_time = datetime.now() + timedelta(days=1)  # Next day
            
            appointment = db_manager.book_appointment(
                st.session_state.patient_data['id'],
                doctor['id'],
                appointment_time.isoformat()
            )
            
            if appointment:
                st.session_state.conversation_state = ConversationState.COMPLETED
                bot_response = f"Great! I've booked your appointment with {doctor['name']} for {appointment_time.strftime('%Y-%m-%d at %H:%M')}."
            else:
                bot_response = BOT_MESSAGES["booking_failed"]
        else:
            bot_response = "I'm sorry, no doctors are available at the moment."
    
    else:
        bot_response = voice_bot.get_gpt_response(user_text, st.session_state.conversation_history)
    
    # Update conversation history
    st.session_state.conversation_history[-1]["bot"] = bot_response
    
    return bot_response

def audio_frame_callback(frame):
    """Callback for audio frames from webrtc"""
    if st.session_state.is_recording:
        st.session_state.audio_queue.put(frame)

def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🎤",
        layout="wide"
    )
    
    st.title(APP_TITLE)
    st.markdown(APP_DESCRIPTION)
    
    # Sidebar for conversation state
    with st.sidebar:
        st.header("Conversation Status")
        st.write(f"**Current State:** {st.session_state.conversation_state}")
        
        if st.session_state.patient_data:
            st.write("**Patient Info:**")
            st.write(f"Name: {st.session_state.patient_data.get('first_name', '')} {st.session_state.patient_data.get('last_name', '')}")
            st.write(f"DOB: {st.session_state.patient_data.get('dob', '')}")
        
        if st.button("Reset Conversation"):
            st.session_state.conversation_state = ConversationState.GREETING
            st.session_state.conversation_history = []
            st.session_state.patient_data = {}
            st.rerun()
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Voice Interface")
        
        # Voice recording section
        st.subheader("🎤 Voice Input")
        
        # WebRTC audio component
        webrtc_ctx = webrtc.webrtc_streamer(
            key="audio",
            mode=webrtc.StreamingMode.AUDIO_ONLY,
            audio_receiver_size=1024,
            media_stream_constraints={"audio": True},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Voice control buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🎤 Start Recording", key="start_recording", type="primary"):
                st.session_state.is_recording = True
                st.info("Recording... Speak now!")
        
        with col_btn2:
            if st.button("⏹️ Stop Recording", key="stop_recording"):
                st.session_state.is_recording = False
                st.success("Recording stopped!")
                
                # Process the recorded audio
                if webrtc_ctx.audio_receiver:
                    # Get audio frames
                    audio_frames = []
                    try:
                        while True:
                            frame = webrtc_ctx.audio_receiver.get_frame(timeout=1)
                            audio_frames.append(frame)
                    except:
                        pass
                    
                    if audio_frames:
                        # Convert frames to audio data
                        audio_data = b''.join([frame.to_ndarray().tobytes() for frame in audio_frames])
                        
                        # Process the audio
                        with st.spinner("Processing your voice..."):
                            bot_response = process_voice_input(audio_data)
                        
                        # Display response
                        st.write("**Bot Response:**")
                        st.write(bot_response)
                        
                        # Generate and play TTS
                        voice_bot = VoiceBot()
                        tts_audio = voice_bot.text_to_speech(bot_response)
                        
                        if tts_audio:
                            # Convert to base64 for audio playback
                            audio_base64 = base64.b64encode(tts_audio).decode()
                            st.audio(f"data:audio/mp3;base64,{audio_base64}", format="audio/mp3")
        
        # Alternative text input for testing
        st.subheader("📝 Text Input (Alternative)")
        user_text_input = st.text_input("Type your message here for testing:")
        
        if user_text_input and st.button("Send Text Message"):
            bot_response = process_voice_input(user_text_input.encode())
            st.write("**Bot Response:**")
            st.write(bot_response)
            
            # Generate TTS
            voice_bot = VoiceBot()
            tts_audio = voice_bot.text_to_speech(bot_response)
            
            if tts_audio:
                audio_base64 = base64.b64encode(tts_audio).decode()
                st.audio(f"data:audio/mp3;base64,{audio_base64}", format="audio/mp3")
    
    with col2:
        st.header("Conversation History")
        
        if st.session_state.conversation_history:
            for i, entry in enumerate(st.session_state.conversation_history[-5:]):
                st.write(f"**User {i+1}:** {entry['user']}")
                st.write(f"**Bot {i+1}:** {entry['bot']}")
                st.write("---")
        else:
            st.write("No conversation yet. Click the microphone to start!")
        
        # Show current state info
        st.subheader("Current State Info")
        st.write(f"**State:** {st.session_state.conversation_state}")
        st.write(f"**Recording:** {st.session_state.is_recording}")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit, OpenAI GPT-4, Whisper, and Supabase")
    
    # Instructions
    st.markdown("### Instructions:")
    st.markdown("""
    1. **Click 'Start Recording'** to begin voice input
    2. **Speak clearly** into your microphone
    3. **Click 'Stop Recording'** when finished
    4. The bot will process your speech and respond
    5. **For appointment booking**, provide your DOB and first 3 letters of first/last name
    6. Use **'Reset Conversation'** to start over
    """)

if __name__ == "__main__":
    main()
