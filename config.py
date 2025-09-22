import os
from dotenv import load_dotenv

load_dotenv()

# API Keys - Load from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# App Configuration
APP_TITLE = "Medical Appointment Voicebot"
APP_DESCRIPTION = "Book your medical appointments using voice commands"

class ConversationState:
    GREETING = "greeting"
    GENERAL_QA = "general_qa"
    VERIFICATION = "verification"
    FALLBACK = "fallback"
    BOOKING = "booking"
    COMPLETED = "completed"

BOT_MESSAGES = {
    "greeting": "Hi! I can help you book a medical appointment. What's your question or do you want to book right away?",
    "verification_start": "To book an appointment, I need to verify your identity. Please provide your date of birth, first 3 letters of your first name, and first 3 letters of your last name.",
    "verification_failed": "I couldn't find a match. Could you please spell out your first name and last name?",
    "booking_success": "Great! I've found an available doctor and booked your appointment.",
    "booking_failed": "I'm sorry, there seems to be an issue booking your appointment. Please try again later."
}