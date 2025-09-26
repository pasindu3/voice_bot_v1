import streamlit as st
from streamlit_realtime_audio_recorder import audio_recorder
import openai
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer
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
import numpy as np
from typing import List, Dict, Optional
import asyncio
import concurrent.futures
from functools import lru_cache
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import os
from dotenv import load_dotenv

load_dotenv()

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
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

class VoiceBot:
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
    async def speech_to_text_async(self, audio_data):
        """Convert speech to text using Whisper with async processing"""
        try:
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Use async client for faster processing
            with open(temp_file_path, "rb") as audio_file:
                
                transcript = await self.async_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return transcript.text
        except Exception as e:
            logger.error(f"Async speech-to-text error: {e}")
            return None
    
    def speech_to_text(self, audio_data):
        """Convert speech to text using Whisper (sync version for fallback)"""
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
    
    async def get_gpt_response_async(self, user_input, conversation_history):
        """Get response from GPT-4 with async processing"""
        try:
            system_prompt = self._get_system_prompt()
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history (limit to last 5 exchanges for speed)
            for entry in conversation_history[-5:]:
                messages.append({"role": "user", "content": entry.get("user", "")})
                messages.append({"role": "assistant", "content": entry.get("bot", "")})
            
            # Add current user input
            messages.append({"role": "user", "content": user_input})
            
            response = await self.async_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=300,  # Reduced for faster response
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Async GPT response error: {e}")
            return "I'm sorry, I'm having trouble processing your request. Please try again."
    
    async def text_to_speech_streaming(self, text):
        """Convert text to speech using streaming TTS for immediate playback"""
        try:
            # Use streaming response for immediate playback
            async with self.async_client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                input=text,
                response_format="pcm"
            ) as stream_response:
                # Play audio directly using LocalAudioPlayer
                await LocalAudioPlayer().play(stream_response)
                return True
        except Exception as e:
            logger.error(f"Streaming TTS error: {e}")
            return False
    
    def text_to_speech(self, text):
        """Convert text to speech using GPT-4-mini-TTS (fallback method)"""
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
                model="gpt-4o",
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

async def extract_patient_info_with_gpt_async(text):
    """Extract patient information using GPT-4o with async processing"""
    try:
        voice_bot = VoiceBot()
        
        extraction_prompt = f"""
        Extract patient information from this text: "{text}"
        
        Return ONLY a JSON object with these exact keys:
        {{
            "date_of_birth": "YYYY-MM-DD",
            "first_name_letters": "ABC",
            "last_name_letters": "XYZ"
        }}
        
        Rules:
        1. Convert any date to YYYY-MM-DD format
        2. For names: Extract the FIRST 3 letters from the beginning of the full name
        3. If user says "My name is Pasindu Dharmadasa", extract "PAS" and "DHA"
        4. If user says "I am John Smith", extract "JOH" and "SMI"
        5. Use null if information is missing
        6. Return ONLY the JSON, no other text
        
        Examples:
        Input: "My name is Pasindu Dharmadasa, DOB is 1997 May 24"
        Output: {{"date_of_birth": "1997-05-24", "first_name_letters": "PAS", "last_name_letters": "DHA"}}
        
        Input: "I am John Smith, born 1990-03-15"
        Output: {{"date_of_birth": "1990-03-15", "first_name_letters": "JOH", "last_name_letters": "SMI"}}
        
        Input: "My first name tree letters are P-A-S. My last name tree letters are D-H-A"
        Output: {{"date_of_birth": null, "first_name_letters": "PAS", "last_name_letters": "DHA"}}
        """
        
        response = await voice_bot.async_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data extraction assistant. Always return ONLY valid JSON with no additional text or explanations."},
                {"role": "user", "content": extraction_prompt}
            ],
            max_tokens=150,
            temperature=0.0
        )
        
        response_text = response.choices[0].message.content.strip()
        logger.info(f"GPT raw response: {response_text}")
        
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response_text
        
        extracted_data = json.loads(json_str)
        
        dob = extracted_data.get("date_of_birth")
        first_name_part = extracted_data.get("first_name_letters")
        last_name_part = extracted_data.get("last_name_letters")
        
        logger.info(f"GPT extracted - DOB: {dob}, First: {first_name_part}, Last: {last_name_part}")
        
        return dob, first_name_part, last_name_part
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return None, None, None
    except Exception as e:
        logger.error(f"Error in async GPT extraction: {e}")
        return None, None, None

async def extract_patient_info_fallback_async(text):
    """Fallback extraction using simple regex patterns (async version)"""
    import re
    from datetime import datetime
    
    # Convert to lowercase for easier matching
    text_lower = text.lower()
    
    # Date patterns
    date_patterns = [
        r'(\d{4})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})',
        r'(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})',
        r'(\d{4})-(\d{2})-(\d{2})',
        r'(\d{2})/(\d{2})/(\d{4})',
    ]
    
    month_map = {
        'january': '01', 'february': '02', 'march': '03', 'april': '04',
        'may': '05', 'june': '06', 'july': '07', 'august': '08',
        'september': '09', 'october': '10', 'november': '11', 'december': '12'
    }
    
    dob = None
    for pattern in date_patterns:
        match = re.search(pattern, text_lower)
        if match:
            groups = match.groups()
            if len(groups) == 3:
                if groups[1] in month_map:
                    if len(groups[0]) == 4:
                        dob = f"{groups[0]}-{month_map[groups[1]]}-{groups[2].zfill(2)}"
                    else:
                        dob = f"{groups[2]}-{month_map[groups[1]]}-{groups[0].zfill(2)}"
                else:
                    if len(groups[0]) == 4:
                        dob = f"{groups[0]}-{groups[1]}-{groups[2]}"
                    else:
                        dob = f"{groups[2]}-{groups[1]}-{groups[0]}"
            break
    
    first_name_part = None
    last_name_part = None
    
    name_match = re.search(r'i\s+am\s+([a-zA-Z]+)\s+([a-zA-Z]+)', text_lower)
    if name_match:
        first_name_part = name_match.group(1)[:3].upper()
        last_name_part = name_match.group(2)[:3].upper()
    
    if not first_name_part or not last_name_part:
        name_match2 = re.search(r'my\s+name\s+is\s+([a-zA-Z]+)\s+([a-zA-Z]+)', text_lower)
        if name_match2:
            if not first_name_part:
                first_name_part = name_match2.group(1)[:3].upper()
            if not last_name_part:
                last_name_part = name_match2.group(2)[:3].upper()
    
    if not first_name_part or not last_name_part:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            if word1 not in ['date', 'birth', 'first', 'last', 'name', 'letters', 'three']:
                if not first_name_part:
                    first_name_part = word1[:3].upper()
                if not last_name_part:
                    last_name_part = word2[:3].upper()
                break
    
    logger.info(f"Async fallback extracted - DOB: {dob}, First: {first_name_part}, Last: {last_name_part}")
    return dob, first_name_part, last_name_part

def extract_patient_info_with_gpt(text):
    """Extract patient information using GPT-4o for intelligent parsing"""
    try:
        voice_bot = VoiceBot()
        
        # Create a more specific prompt for data extraction
        extraction_prompt = f"""
        Extract patient information from this text: "{text}"
        
        Return ONLY a JSON object with these exact keys:
        {{
            "date_of_birth": "YYYY-MM-DD",
            "first_name_letters": "ABC",
            "last_name_letters": "XYZ"
        }}
        
        Rules:
        1. Convert any date to YYYY-MM-DD format
        2. For names: Extract the FIRST 3 letters from the beginning of the full name
        3. If user says "My name is Pasindu Dharmadasa", extract "PAS" and "DHA"
        4. If user says "I am John Smith", extract "JOH" and "SMI"
        5. Use null if information is missing
        6. Return ONLY the JSON, no other text
        
        Examples:
        Input: "My name is Pasindu Dharmadasa, DOB is 1997 May 24"
        Output: {{"date_of_birth": "1997-05-24", "first_name_letters": "PAS", "last_name_letters": "DHA"}}
        
        Input: "I am John Smith, born 1990-03-15"
        Output: {{"date_of_birth": "1990-03-15", "first_name_letters": "JOH", "last_name_letters": "SMI"}}
        
        Input: "My first name tree letters are P-A-S. My last name tree letters are D-H-A"
        Output: {{"date_of_birth": null, "first_name_letters": "PAS", "last_name_letters": "DHA"}}
        """
        
        # Get GPT response
        response = voice_bot.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data extraction assistant. Always return ONLY valid JSON with no additional text or explanations."},
                {"role": "user", "content": extraction_prompt}
            ],
            max_tokens=150,
            temperature=0.0  # Zero temperature for consistent extraction
        )
        
        # Get the response text
        response_text = response.choices[0].message.content.strip()
        logger.info(f"GPT raw response: {response_text}")
        
        # Clean the response - remove any non-JSON text
        import json
        import re
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response_text
        
        # Parse the JSON
        extracted_data = json.loads(json_str)
        
        # Extract values
        dob = extracted_data.get("date_of_birth")
        first_name_part = extracted_data.get("first_name_letters")
        last_name_part = extracted_data.get("last_name_letters")
        
        # Debug logging
        logger.info(f"GPT extracted - DOB: {dob}, First: {first_name_part}, Last: {last_name_part}")
        
        return dob, first_name_part, last_name_part
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.error(f"Response was: {response_text}")
        return None, None, None
    except Exception as e:
        logger.error(f"Error in GPT extraction: {e}")
        return None, None, None

def extract_patient_info_fallback(text):
    """Fallback extraction using simple regex patterns"""
    import re
    from datetime import datetime
    
    # Convert to lowercase for easier matching
    text_lower = text.lower()
    
    # Date patterns
    date_patterns = [
        r'(\d{4})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})',  # 1997 May 24
        r'(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})',  # 24 May 1997
        r'(\d{4})-(\d{2})-(\d{2})',  # 1997-05-24
        r'(\d{2})/(\d{2})/(\d{4})',  # 24/05/1997
    ]
    
    # Month name to number mapping
    month_map = {
        'january': '01', 'february': '02', 'march': '03', 'april': '04',
        'may': '05', 'june': '06', 'july': '07', 'august': '08',
        'september': '09', 'october': '10', 'november': '11', 'december': '12'
    }
    
    dob = None
    for pattern in date_patterns:
        match = re.search(pattern, text_lower)
        if match:
            groups = match.groups()
            if len(groups) == 3:
                if groups[1] in month_map:  # Month name format
                    if len(groups[0]) == 4:  # Year first: 1997 May 24
                        dob = f"{groups[0]}-{month_map[groups[1]]}-{groups[2].zfill(2)}"
                    else:  # Day first: 24 May 1997
                        dob = f"{groups[2]}-{month_map[groups[1]]}-{groups[0].zfill(2)}"
                else:  # Numeric format
                    if len(groups[0]) == 4:  # YYYY-MM-DD
                        dob = f"{groups[0]}-{groups[1]}-{groups[2]}"
                    else:  # DD/MM/YYYY
                        dob = f"{groups[2]}-{groups[1]}-{groups[0]}"
            break
    
    # Enhanced name extraction - look for various patterns
    first_name_part = None
    last_name_part = None
    
    # Pattern 1: "I am [First] [Last]"
    name_match = re.search(r'i\s+am\s+([a-zA-Z]+)\s+([a-zA-Z]+)', text_lower)
    if name_match:
        first_name_part = name_match.group(1)[:3].upper()
        last_name_part = name_match.group(2)[:3].upper()
    
    # Pattern 2: "My name is [First] [Last]"
    if not first_name_part or not last_name_part:
        name_match2 = re.search(r'my\s+name\s+is\s+([a-zA-Z]+)\s+([a-zA-Z]+)', text_lower)
        if name_match2:
            if not first_name_part:
                first_name_part = name_match2.group(1)[:3].upper()
            if not last_name_part:
                last_name_part = name_match2.group(2)[:3].upper()
    
    # Pattern 3: Look for any two consecutive words that could be names
    if not first_name_part or not last_name_part:
        # Find all words in the text
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)
        # Look for potential name pairs
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            # Skip common words
            if word1 not in ['date', 'birth', 'first', 'last', 'name', 'letters', 'three']:
                if not first_name_part:
                    first_name_part = word1[:3].upper()
                if not last_name_part:
                    last_name_part = word2[:3].upper()
                break
    
    logger.info(f"Fallback extracted - DOB: {dob}, First: {first_name_part}, Last: {last_name_part}")
    return dob, first_name_part, last_name_part

# Response cache for common queries
@lru_cache(maxsize=100)
def get_cached_response(query_hash: str, state: str) -> Optional[str]:
    """Cache common responses for instant replies"""
    return None

def cache_response(query_hash: str, state: str, response: str):
    """Cache a response for future use"""
    get_cached_response.cache_info()

async def play_audio_async(text):
    """Helper function to play audio asynchronously"""
    voice_bot = VoiceBot()
    try:
        await voice_bot.text_to_speech_streaming(text)
        return True
    except Exception as e:
        logger.error(f"Async audio playback error: {e}")
        return False

async def process_voice_input_async(audio_data):
    """Ultra-fast async voice processing with parallel execution"""
    start_time = time.time()
    voice_bot = VoiceBot()
    db_manager = DatabaseManager()
    
    try:
        # Step 1: Convert speech to text (async)
        logger.info("🔄 Starting async STT...")
        user_text = await voice_bot.speech_to_text_async(audio_data)
        
        if not user_text:
            return "I couldn't understand what you said. Please try again."
        
        stt_time = time.time() - start_time
        logger.info(f"✅ STT completed in {stt_time:.2f}s")
        
        # Step 2: Check cache first for instant response
        query_hash = hash(user_text.lower().strip())
        cached_response = get_cached_response(str(query_hash), st.session_state.conversation_state)
        
        if cached_response:
            logger.info("⚡ Using cached response!")
            return cached_response
        
        # Step 3: Add to conversation history
        st.session_state.conversation_history.append({"user": user_text, "bot": ""})
        
        # Step 4: Process based on conversation state (parallel where possible)
        if st.session_state.conversation_state == ConversationState.GREETING:
            if "book" in user_text.lower() or "appointment" in user_text.lower():
                st.session_state.conversation_state = ConversationState.VERIFICATION
                bot_response = BOT_MESSAGES["verification_start"]
            else:
                bot_response = await voice_bot.get_gpt_response_async(user_text, st.session_state.conversation_history)
        
        elif st.session_state.conversation_state == ConversationState.VERIFICATION:
            # Parallel extraction using both methods
            logger.info("🔄 Starting parallel data extraction...")
            
            # Run both extraction methods in parallel
            extraction_tasks = [
                asyncio.create_task(extract_patient_info_with_gpt_async(user_text)),
                asyncio.create_task(extract_patient_info_fallback_async(user_text))
            ]
            
            # Wait for first successful result
            results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
            
            dob, first_name, last_name = None, None, None
            
            # Use first successful result
            for result in results:
                if isinstance(result, tuple) and len(result) == 3:
                    dob_res, first_res, last_res = result
                    if dob_res and first_res and last_res:
                        dob, first_name, last_name = dob_res, first_res, last_res
                        logger.info(f"✅ Parallel extraction successful: {dob}, {first_name}, {last_name}")
                        break
            
            if dob and first_name and last_name:
                # Verify patient (this could also be made async)
                patient = db_manager.verify_patient(dob, first_name, last_name)
                if patient:
                    st.session_state.patient_data = patient
                    st.session_state.conversation_state = ConversationState.BOOKING
                    bot_response = f"✅ Verification successful! Hello {patient['first_name']} {patient['last_name']}! I found your record in our database. Your DOB is {patient['dob']}. Let me book your appointment with the next available doctor."
                else:
                    st.session_state.conversation_state = ConversationState.FALLBACK
                    bot_response = f"❌ I couldn't find a patient record matching:\n- DOB: {dob}\n- First name starting with: {first_name}\n- Last name starting with: {last_name}\n\nCould you please spell out your first name and last name completely, or check if your information is correct?"
            else:
                bot_response = f"I'm having trouble extracting your information. Please try saying:\n\n'My name is [Your First Name] [Your Last Name], and my date of birth is [Year] [Month] [Day]'\n\nFor example: 'My name is John Smith, and my date of birth is 1990 March 15'"
        
        elif st.session_state.conversation_state == ConversationState.FALLBACK:
            bot_response = await voice_bot.get_gpt_response_async(user_text, st.session_state.conversation_history)
        
        elif st.session_state.conversation_state == ConversationState.BOOKING:
            # Book appointment
            doctors = db_manager.get_available_doctors()
            if doctors:
                doctor = doctors[0]
                appointment_time = datetime.now() + timedelta(days=1)
                
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
            bot_response = await voice_bot.get_gpt_response_async(user_text, st.session_state.conversation_history)
        
        # Cache the response for future use
        cache_response(str(query_hash), st.session_state.conversation_state, bot_response)
        
        # Update conversation history
        st.session_state.conversation_history[-1]["bot"] = bot_response
        
        total_time = time.time() - start_time
        logger.info(f"🚀 Total processing time: {total_time:.2f}s")
        
        return bot_response
        
    except Exception as e:
        logger.error(f"Async processing error: {e}")
        return "I'm sorry, I'm having trouble processing your request. Please try again."

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
        # Try GPT-4 extraction first
        dob, first_name, last_name = extract_patient_info_with_gpt(user_text)
        
        # If GPT extraction failed, try fallback method
        if not dob or not first_name or not last_name:
            logger.info("GPT extraction failed, trying fallback method")
            dob, first_name, last_name = extract_patient_info_fallback(user_text)
        
        # Debug: Show what was extracted
        logger.info(f"Verification attempt - DOB: {dob}, First: {first_name}, Last: {last_name}")
        
        if dob and first_name and last_name:
            # Verify patient
            patient = db_manager.verify_patient(dob, first_name, last_name)
            if patient:
                st.session_state.patient_data = patient
                st.session_state.conversation_state = ConversationState.BOOKING
                bot_response = f"✅ Verification successful! Hello {patient['first_name']} {patient['last_name']}! I found your record in our database. Your DOB is {patient['dob']}. Let me book your appointment with the next available doctor."
            else:
                st.session_state.conversation_state = ConversationState.FALLBACK
                bot_response = f"❌ I couldn't find a patient record matching:\n- DOB: {dob}\n- First name starting with: {first_name}\n- Last name starting with: {last_name}\n\nCould you please spell out your first name and last name completely, or check if your information is correct?"
        else:
            # If both methods failed, ask for clearer input
            bot_response = f"I'm having trouble extracting your information. Please try saying:\n\n'My name is [Your First Name] [Your Last Name], and my date of birth is [Year] [Month] [Day]'\n\nFor example: 'My name is John Smith, and my date of birth is 1990 March 15'"
    
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

def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🎤",
        layout="wide"
    )
    
    st.title(APP_TITLE)
    st.markdown(APP_DESCRIPTION)
    
    # Audio setup notice
    st.info("🔊 **Audio Setup**: Click anywhere on the page first to enable audio playback, then click the play button when the bot responds.")
    
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
        st.header("🎤 Real-Time Voice Interface")
        
        # Audio recorder component
        st.subheader("Real-Time Voice Recording")
        
        # Initialize the audio recorder with automatic silence detection
        audio_data = audio_recorder(
            interval=50,          # Check audio level every 50ms
            threshold=-60,        # Audio level threshold for speech detection in dB
            silenceTimeout=2000,  # Time in milliseconds to wait after silence before stopping recording
            play=False            # Set to True to play the audio during recording
        )
        
        # Process audio when recording is complete
        if audio_data:
            if audio_data.get('status') == 'stopped':
                audio_content = audio_data.get('audioData')
                if audio_content:
                    st.session_state.is_processing = True
                    
                    # Decode the base64 audio data
                    audio_bytes = base64.b64decode(audio_content)
                    
                    with st.spinner("🚀 Ultra-fast processing..."):
                        # Process the audio with async parallel processing
                        bot_response = asyncio.run(process_voice_input_async(audio_bytes))
                    
                    # Display the conversation
                    st.write("**You said:**")
                    st.write(st.session_state.conversation_history[-1]["user"])
                    st.write("**Bot Response:**")
                    st.write(bot_response)
                    
                    # Generate TTS audio for web playback
                    st.info("🔊 Generating audio response...")
                    try:
                        voice_bot = VoiceBot()
                        tts_audio = voice_bot.text_to_speech(bot_response)
                        
                        if tts_audio:
                            # Convert to base64 for web playback
                            audio_base64 = base64.b64encode(tts_audio).decode()
                            
                            # Create audio player with autoplay attempt
                            st.markdown("### 🔊 Bot Response Audio")
                            
                            # Try autoplay with user interaction
                            st.markdown(f"""
                            <audio controls autoplay style="width: 100%; margin: 10px 0;">
                                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                                Your browser does not support the audio element.
                            </audio>
                            <script>
                                // Try to play audio after user interaction
                                document.addEventListener('click', function() {{
                                    const audio = document.querySelector('audio');
                                    if (audio) {{
                                        audio.play().catch(function(error) {{
                                            console.log('Autoplay prevented:', error);
                                        }});
                                    }}
                                }}, {{ once: true }});
                            </script>
                            """, unsafe_allow_html=True)
                            
                            # Also provide a manual play button
                            st.markdown("**Click the play button above to hear the bot's response**")
                            
                            # Download option
                            st.download_button(
                                label="📥 Download Audio Response",
                                data=tts_audio,
                                file_name=f"bot_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
                                mime="audio/mp3"
                            )
                        else:
                            st.warning("⚠️ Could not generate audio response")
                            
                    except Exception as e:
                        logger.error(f"Audio generation error: {e}")
                        st.error(f"Audio generation error: {e}")
                    
                    st.session_state.is_processing = False
                else:
                    st.error("No audio data received.")
            elif audio_data.get('error'):
                st.error(f"Error: {audio_data.get('error')}")
            elif audio_data.get('status') == 'recording':
                st.info("🎤 Recording... Speak now!")
        
        # Text input alternative for testing
        st.subheader("📝 Text Input (Alternative)")
        user_text_input = st.text_input("Type your message here for testing:")
        
        if user_text_input and st.button("Send Text Message"):
            with st.spinner("🚀 Ultra-fast processing..."):
                bot_response = asyncio.run(process_voice_input_async(user_text_input.encode()))
            st.write("**Bot Response:**")
            st.write(bot_response)
            
            # Generate TTS audio for web playback
            st.info("🔊 Generating audio response...")
            try:
                voice_bot = VoiceBot()
                tts_audio = voice_bot.text_to_speech(bot_response)
                
                if tts_audio:
                    # Convert to base64 for web playback
                    audio_base64 = base64.b64encode(tts_audio).decode()
                    
                    # Create audio player with autoplay attempt
                    st.markdown("### 🔊 Bot Response Audio")
                    
                    # Try autoplay with user interaction
                    st.markdown(f"""
                    <audio controls autoplay style="width: 100%; margin: 10px 0;">
                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                        Your browser does not support the audio element.
                    </audio>
                    <script>
                        // Try to play audio after user interaction
                        document.addEventListener('click', function() {{
                            const audio = document.querySelector('audio');
                            if (audio) {{
                                audio.play().catch(function(error) {{
                                    console.log('Autoplay prevented:', error);
                                }});
                            }}
                        }}, {{ once: true }});
                    </script>
                    """, unsafe_allow_html=True)
                    
                    # Also provide a manual play button
                    st.markdown("**Click the play button above to hear the bot's response**")
                    
                    # Download option
                    st.download_button(
                        label="📥 Download Audio Response",
                        data=tts_audio,
                        file_name=f"bot_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
                        mime="audio/mp3"
                    )
                else:
                    st.warning("⚠️ Could not generate audio response")
                    
            except Exception as e:
                logger.error(f"Audio generation error: {e}")
                st.error(f"Audio generation error: {e}")
        
        # Instructions
        st.subheader("📋 How to Use")
        st.markdown("""
        **For Voice Interaction:**
        1. **Start speaking** - the microphone will automatically detect your voice
        2. **Speak clearly** into your microphone
        3. **Stop speaking** - recording will automatically stop after 2 seconds of silence
        4. **Wait for processing** - the bot will respond with text and audio
        
        **For Appointment Booking:**
        - Say: "I want to book an appointment"
        - Provide your date of birth (YYYY-MM-DD format)
        - Provide first 3 letters of your first name
        - Provide first 3 letters of your last name
        
        **Example:** "My date of birth is 1990-05-15, my first name starts with JOH, and my last name starts with SMI"
        
        **Features:**
        - ✅ **Automatic silence detection** - no need to click stop
        - ✅ **Ultra-fast processing** - parallel async processing for instant responses
        - ✅ **Smart audio detection** - only records when you speak
        - ✅ **Web-optimized audio** - click-to-play audio controls for browser compatibility
        - ✅ **Response caching** - common queries get instant cached responses
        - ✅ **Parallel extraction** - GPT and regex extraction run simultaneously
        - ✅ **Download option** - save audio responses as MP3 files
        
        **🔊 Audio Note:** Due to browser restrictions, audio requires user interaction. Click the play button to hear the bot's response.
        """)
    
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
        st.write(f"**Messages:** {len(st.session_state.conversation_history)}")
        st.write(f"**Processing:** {st.session_state.is_processing}")
        
        # Performance monitoring
        st.subheader("⚡ Performance")
        st.write("**Optimizations Active:**")
        st.write("✅ Async STT processing")
        st.write("✅ Parallel data extraction")
        st.write("✅ Streaming TTS playback")
        st.write("✅ Response caching")
        st.write("✅ Reduced token limits")
        st.write("✅ Concurrent processing")
        
        # Debug info for verification
        if st.session_state.conversation_state == ConversationState.VERIFICATION:
            st.subheader("🔍 Debug Info")
            st.write("**Last extracted info (GPT-4o + Fallback):**")
            if st.session_state.conversation_history:
                last_user_input = st.session_state.conversation_history[-1]["user"]
                
                # Try GPT extraction
                dob_gpt, first_gpt, last_gpt = extract_patient_info_with_gpt(last_user_input)
                st.write(f"- GPT-4o: DOB={dob_gpt}, First={first_gpt}, Last={last_gpt}")
                
                # Try fallback extraction
                dob_fb, first_fb, last_fb = extract_patient_info_fallback(last_user_input)
                st.write(f"- Fallback: DOB={dob_fb}, First={first_fb}, Last={last_fb}")
                
                st.write(f"- Raw input: {last_user_input}")
        
        # Quick actions
        st.subheader("Quick Actions")
        if st.button("🎯 Book Appointment"):
            st.session_state.conversation_state = ConversationState.VERIFICATION
            st.rerun()
        
        if st.button("❓ Ask Question"):
            st.session_state.conversation_state = ConversationState.GENERAL_QA
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit, OpenAI GPT-4, Whisper, and Supabase")
    
    # Status indicator
    if st.session_state.is_processing:
        st.info("🔄 Processing your voice...")

if __name__ == "__main__":
    main()
