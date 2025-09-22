# 🎤 Medical Appointment Voicebot

A real-time voice-powered medical appointment booking system built with Streamlit, OpenAI GPT-4, Whisper, and Supabase.

## ✨ Features

- 🎤 **Real-time Voice Processing** - Speak naturally to interact with the bot
- ⚡ **Ultra-fast Response** - Parallel async processing for instant responses
- 🔊 **Instant Audio Playback** - Streaming TTS for immediate voice feedback
- 🏥 **Patient Verification** - Secure identity verification using partial name matching
- 📅 **Appointment Booking** - Automatic booking with next available doctor
- 💾 **Response Caching** - Instant replies for common queries
- 🎯 **Smart Data Extraction** - GPT-4 powered intelligent data parsing

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI Models**: OpenAI GPT-4, Whisper, TTS-1
- **Database**: Supabase
- **Audio Processing**: streamlit-realtime-audio-recorder
- **Performance**: AsyncIO, Parallel Processing, Response Caching

## 📋 Prerequisites

- Python 3.12+
- OpenAI API Key
- Supabase Account
- Microphone access

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/medical-voicebot.git
   cd medical-voicebot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   ```

4. **Run the application**
   ```bash
   streamlit run voice_app_realtime.py
   ```

## 🎯 Usage

### Voice Interaction
1. Click the microphone button
2. Speak your request naturally
3. The bot will process and respond with voice

### Appointment Booking
1. Say "I want to book an appointment"
2. Provide your date of birth
3. Give first 3 letters of your first name
4. Give first 3 letters of your last name
5. Bot will verify and book your appointment

### Example Commands
- "I want to book an appointment"
- "My name is John Smith, DOB is 1990-05-15"
- "What are your office hours?"
- "How do I cancel an appointment?"

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Voice Input   │───▶│  Speech-to-Text  │───▶│   GPT-4 Processing │
│  (Microphone)   │    │    (Whisper)     │    │   (Async)        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Audio Output   │◀───│  Text-to-Speech  │◀───│  Response Cache │
│  (Streaming)    │    │   (TTS-1)       │    │   (LRU Cache)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Database      │◀───│ Patient Verification │◀───│ Data Extraction │
│   (Supabase)    │    │   (Parallel)     │    │   (GPT + Regex) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## ⚡ Performance Optimizations

- **Async Processing**: All API calls are asynchronous
- **Parallel Extraction**: GPT-4 and regex extraction run simultaneously
- **Streaming TTS**: Audio plays while TTS generates
- **Response Caching**: Common queries get instant responses
- **Reduced Tokens**: Optimized for speed
- **Concurrent Processing**: Multiple operations run in parallel

## 🔒 Security

- Environment variables for sensitive data
- No hardcoded API keys
- Secure patient verification
- Data privacy compliance

## 📊 Database Schema

### Patients Table
```sql
CREATE TABLE patients (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    dob DATE,
    email VARCHAR(255),
    phone VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Appointments Table
```sql
CREATE TABLE appointments (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),
    doctor_id INTEGER,
    appointment_time TIMESTAMP,
    status VARCHAR(50) DEFAULT 'scheduled',
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 🚀 Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Connect your GitHub repository to Streamlit Cloud
3. Add environment variables in Streamlit Cloud settings
4. Deploy!

### Local Deployment
```bash
streamlit run voice_app_realtime.py --server.port 8501
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide

## 🎉 Acknowledgments

- OpenAI for GPT-4 and Whisper
- Streamlit for the amazing framework
- Supabase for the database backend
- The open-source community

---

**Built with ❤️ for better healthcare accessibility**