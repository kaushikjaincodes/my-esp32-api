from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
import speech_recognition as sr
import google.generativeai as genai
from gtts import gTTS
from dotenv import load_dotenv
import tempfile
import os
import io
import wave
from pydub import AudioSegment

load_dotenv()
app = FastAPI()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash")

# ===============================================
# ✅ Main endpoint: PCM → WAV → STT → Gemini → WAV
# ===============================================
@app.post("/")
async def process_audio(request: Request):
    try:
        raw_pcm = await request.body()

        # ---------------------------
        # 1️⃣ Save raw PCM → WAV file
        # ---------------------------
        pcm_wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with wave.open(pcm_wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(raw_pcm)

        # ---------------------------
        # 2️⃣ Speech-to-Text
        # ---------------------------
        recognizer = sr.Recognizer()
        with sr.AudioFile(pcm_wav_path) as source:
            audio = recognizer.record(source)

        try:
            text_input = recognizer.recognize_google(audio)
        except:
            text_input = "[Could not understand]"
        print("Recognized:", text_input)

        # ---------------------------
        # 3️⃣ Gemini response
        # ---------------------------
        prompt = f"Reply in less than 30 words: {text_input}"
        response = model.generate_content(prompt)
        ai_text = response.text.strip()
        print("Gemini:", ai_text)

        # ---------------------------
        # 4️⃣ Text → speech (MP3)
        # ---------------------------
        tts = gTTS(ai_text)
        mp3_bytes = io.BytesIO()
        tts.write_to_fp(mp3_bytes)
        mp3_bytes.seek(0)

        # ---------------------------
        # 5️⃣ Convert MP3 → WAV (16kHz mono 16-bit)
        # ---------------------------
        audio = AudioSegment.from_file(mp3_bytes, format="mp3")
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)

        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        print("✅ Sending WAV back to ESP32")

        return Response(content=wav_io.read(), media_type="audio/wav")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
