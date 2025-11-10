from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
import speech_recognition as sr
import google.generativeai as genai
from dotenv import load_dotenv
import tempfile
import io
import os
import wave
from gtts import gTTS
from pydub import AudioSegment

# --- Setup ---
load_dotenv()
app = FastAPI(title="ESP32 ↔ Gemini Voice Assistant (WAV Mode)")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
client = genai.GenerativeModel("models/gemini-2.5-flash")

@app.post("/")
async def full_voice_pipeline(request: Request):
    try:
        raw_pcm = await request.body()
        print(f"🎧 Received {len(raw_pcm)} bytes of PCM data")

        # --- 1️⃣ PCM → WAV ---
        wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)              # Mono
            wf.setsampwidth(2)              # 16-bit PCM
            wf.setframerate(16000)
            wf.writeframes(raw_pcm)
        print(f"✅ Saved PCM as WAV: {wav_path}")

        # --- 2️⃣ STT ---
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            try:
                text_input = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                text_input = "[Could not understand speech]"
        print(f"🗣️ Recognized: {text_input}")

        # --- 3️⃣ Gemini ---
        prompt = f"Reply in less than 30 words: {text_input}"
        response = client.generate_content(prompt)
        ai_reply = getattr(response, "text", str(response)).strip()
        print(f"🤖 Gemini Reply: {ai_reply}")

        # --- 4️⃣ Convert text → speech (WAV) ---
        tts = gTTS(ai_reply, lang='en')
        mp3_bytes = io.BytesIO()
        tts.write_to_fp(mp3_bytes)
        mp3_bytes.seek(0)

        # Convert MP3 → WAV (16-bit, 16kHz, mono)
        sound = AudioSegment.from_file(mp3_bytes, format="mp3")
        wav_io = io.BytesIO()
        sound = sound.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        sound.export(wav_io, format="wav")
        wav_io.seek(0)

        print("✅ Returning WAV audio response")
        return Response(content=wav_io.read(), media_type="audio/wav")

    except Exception as e:
        print(f"❌ Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
