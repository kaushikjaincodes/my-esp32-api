from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response, JSONResponse

import speech_recognition as sr
from google import genai
from gtts import gTTS
from dotenv import load_dotenv
import tempfile
import io
import os
from pydantic import BaseModel
from typing import List, Dict

# --- Setup ---
load_dotenv()
app = FastAPI(title="Gemini API Emulator for ESP32")

# Configure your Gemini client
# Make sure your GOOGLE_API_KEY is in your .env file
# genai.configure() will be called implicitly by genai.Client()
client = genai.Client() # Using the syntax you requested


# --- Pydantic Models (to understand Arduino's JSON) ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    model: str
    messages: List[ChatMessage]

class TTSPayload(BaseModel):
    model: str
    input: str
    voice: str


# --- 1. Speech-to-Text (STT) Endpoint ---
@app.post("/v1/audio/transcriptions")
async def speech_to_text(
    file: UploadFile = File(...), 
    model: str = Form("whisper-1") # We ignore the model, but it's part of the form
):
    """
    Receives audio file, returns transcribed text.
    Matches: gptChat.speechToTextFromBuffer()
    """
    print("🎤 /v1/audio/transcriptions endpoint hit")
    recognizer = sr.Recognizer()
    temp_audio_path = None
    try:
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio_path = temp_audio.name
            temp_audio.write(await file.read())
            temp_audio.flush()
        
        # Recognize speech from the file
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
            text_input = recognizer.recognize_google(audio_data)
        
        print(f"    -> Recognized: {text_input}")
        # Return in the exact format the Arduino expects
        return JSONResponse(content={"text": text_input})

    except sr.UnknownValueError:
        print("    -> Error: Speech could not be understood")
        return JSONResponse({"error": "Speech could not be understood."}, status_code=400)
    except sr.RequestError as e:
        print(f"    -> Error: Speech recognition API error: {e}")
        return JSONResponse({"error": f"Speech recognition API error: {e}"}, status_code=500)
    except Exception as e:
        print(f"    -> Error: Internal Server Error: {e}")
        return JSONResponse({"error": f"Internal Server Error: {str(e)}"}, status_code=500)
    finally:
        # Clean up the temporary file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


# --- 2. Chat (LLM) Endpoint ---
@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatPayload):
    """
    Receives a chat history, sends last message to Gemini, returns response.
    Matches: gptChat.sendMessage()
    """
    print("🤖 /v1/chat/completions endpoint hit")
    try:
        # Extract the system prompt (if any)
        system_prompt = "You are a helpful assistant."
        for msg in payload.messages:
            if msg.role == "system":
                system_prompt = msg.content
                break
        
        # Extract the user's prompt (the last message)
        user_prompt = "No user message found."
        if payload.messages and payload.messages[-1].role == "user":
            user_prompt = payload.messages[-1].content
        
        print(f"    -> User Prompt: {user_prompt}")
        
        full_prompt = f"{system_prompt}\n\nUser: {user_prompt}"

        # --- MODIFIED SECTION: Using your requested syntax ---
        response = client.models.generate_content(
            model="models/gemini-2.5-flash", # Make sure to use the full model name
            contents=full_prompt
        )
        
        # Use getattr to safely get the text, just like in your original code
        text_output = getattr(response, "text", str(response))
        # --- END OF MODIFIED SECTION ---

        print(f"    -> Gemini Response: {text_output}")

        # Format the response to match OpenAI's structure
        chat_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": text_output
                }
            }]
        }
        return JSONResponse(content=chat_response)

    except Exception as e:
        print(f"    -> Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# --- 3. Text-to-Speech (TTS) Endpoint ---
@app.post("/v1/audio/speech")
async def text_to_speech(payload: TTSPayload):
    """
    Receives text, returns an MP3 audio stream.
    Matches: gptChat.textToSpeech()
    """
    print("🔊 /v1/audio/speech endpoint hit")
    try:
        print(f"    -> Generating speech for: {payload.input}")
        
        # Generate speech using gTTS
        tts = gTTS(text=payload.input, lang="en")
        audio_bytes_io = io.BytesIO()
        tts.write_to_fp(audio_bytes_io)
        audio_bytes_io.seek(0)

        # Return the MP3 data as a response
        return Response(
            content=audio_bytes_io.read(),
            media_type="audio/mpeg"
        )
        
    except Exception as e:
        print(f"    -> Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    # Run on 0.0.0.0 to make it accessible on your local network
    uvicorn.run(app, host="0.0.0.0", port=8000)