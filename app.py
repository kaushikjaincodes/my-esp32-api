from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
import wave, io, os, tempfile
import speech_recognition as sr
from gtts import gTTS
import google.generativeai as genai
from dotenv import load_dotenv
from pydub import AudioSegment
import atexit
import logging

# ---------------------- CONFIG ----------------------
load_dotenv()
app = FastAPI(title="Voice Assistant")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash")

# Keep track of temp files for cleanup
temp_files = []

def cleanup_temp_files():
    """Clean up all temporary files on shutdown"""
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Cleaned up: {file_path}")
        except Exception as e:
            logging.error(f"Failed to clean up {file_path}: {e}")

atexit.register(cleanup_temp_files)

# ---------------------- ENDPOINT ----------------------
@app.post("/upload")
async def upload_audio(request: Request):
    """Receives raw PCM16 audio from ESP32 → STT → Gemini → TTS → returns WAV"""
    wav_path = None
    temp_mp3 = None
    tts_path = None
    
    try:
        raw = await request.body()
        print(f"🎧 Received {len(raw)} bytes of PCM data")
        
        # Validate minimum audio length (at least 0.5 seconds)
        min_bytes = 16000 * 2 * 0.5  # sample_rate * bytes_per_sample * seconds
        if len(raw) < min_bytes:
            return JSONResponse(
                status_code=400,
                content={"error": "Audio too short"}
            )
        
        # ---- Step 1: Save raw → WAV ----
        wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        temp_files.append(wav_path)
        
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(raw)
        
        print(f"✅ Saved temp WAV: {wav_path}")
        
        # ---- Step 2: Speech to Text ----
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300  # Adjust for ESP32 mic sensitivity
        text = ""
        
        try:
            with sr.AudioFile(wav_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                print(f"🗣 Recognized: {text}")
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
            text = "Sorry, I could not understand that."
        except sr.RequestError as e:
            print(f"❌ STT service error: {e}")
            return JSONResponse(
                status_code=503,
                content={"error": "Speech recognition service unavailable"}
            )
        except Exception as e:
            print(f"❌ STT failed: {e}")
            text = "Sorry, I could not understand that."
        
        # ---- Step 3: Gemini Response ----
        try:
            prompt = f"User said: '{text}'\nRespond in less than 30 words, conversational and natural. Be friendly and helpful."
            response = model.generate_content(prompt)
            reply_text = response.text.strip()
            print(f"🤖 Gemini: {reply_text}")
        except Exception as e:
            print(f"❌ Gemini error: {e}")
            reply_text = "I encountered an error. Please try again."
        
        # ---- Step 4: Text to Speech (gTTS) ----
        tts_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        temp_files.append(tts_path)
        
        try:
            tts = gTTS(reply_text, lang="en", slow=False)
            temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            temp_files.append(temp_mp3)
            tts.save(temp_mp3)
            
            # Convert MP3 → WAV (16kHz mono for ESP32 compatibility)
            sound = AudioSegment.from_mp3(temp_mp3)
            sound = sound.set_frame_rate(16000).set_channels(1)
            sound.export(tts_path, format="wav")
            
            print(f"🔊 Generated TTS reply.wav ({os.path.getsize(tts_path)} bytes)")
            
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Text-to-speech generation failed"}
            )
        
        # ---- Step 5: Return WAV ----
        return FileResponse(
            tts_path, 
            media_type="audio/wav", 
            filename="reply.wav",
            headers={
                "Content-Disposition": "attachment; filename=reply.wav"
            }
        )
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/")
async def root():
    return {"status": "Voice Assistant API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "gemini-2.5-flash"}