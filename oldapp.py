from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response, JSONResponse
import speech_recognition as sr
from google import genai
from gtts import gTTS
from dotenv import load_dotenv
import tempfile
import io
import os


load_dotenv()

app = FastAPI(title="Speech-to-Speech Gemini API")


client = genai.Client()

@app.post("/process-audio")
async def process_audio(
    audio: UploadFile = File(...),
    prompt: str = Form("You are a helpful assistant.")
):
    try:
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(await audio.read())
            temp_audio.flush()
            with sr.AudioFile(temp_audio.name) as source:
                audio_data = recognizer.record(source)
                text_input = recognizer.recognize_google(audio_data)

        print(f"🎤 Recognized Speech: {text_input}")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{prompt}\nUser said: {text_input}"
        )

        text_output = getattr(response, "text", str(response))
        print(f"🤖 Gemini Response: {text_output}")


        tts = gTTS(text=text_output, lang="en")
        audio_bytes_io = io.BytesIO()
        tts.write_to_fp(audio_bytes_io)
        audio_bytes_io.seek(0)

     
        return Response(
            content=audio_bytes_io.read(),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=response.mp3"}
        )

    except sr.UnknownValueError:
        return JSONResponse({"error": "Speech could not be understood."}, status_code=400)

    except sr.RequestError as e:
        return JSONResponse({"error": f"Speech recognition API error: {e}"}, status_code=500)

    except Exception as e:
        print(f"❌ Internal Error: {e}")
        return JSONResponse({"error": f"Internal Server Error: {str(e)}"}, status_code=500)

    finally:
        if 'temp_audio' in locals() and os.path.exists(temp_audio.name):
            os.remove(temp_audio.name)
