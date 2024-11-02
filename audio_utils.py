import os
import pygame # type: ignore
import sounddevice as sd # type: ignore
import wave
import tempfile
from langdetect import detect, LangDetectException # type: ignore
from deepgram import Deepgram # type: ignore
import requests

audio_folder = "audio_files"
if not os.path.exists(audio_folder):
    os.makedirs(audio_folder)

pygame.mixer.init()

def clear_audio_files():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
    if os.path.exists(audio_folder):
        for file_name in os.listdir(audio_folder):
            file_path = os.path.join(audio_folder, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return 'en'

def play_audio(text, file_name):
    # Generate audio using Deepgram and play it
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    if not DEEPGRAM_API_KEY:
        raise ValueError("Deepgram API key is not set in the environment variables.")
    
    audio_file_path = os.path.join(audio_folder, f"{file_name}.mp3")
    if not os.path.exists(audio_file_path):
        # Use Deepgram to generate audio
        response = requests.post(
            "https://api.deepgram.com/v1/speak",
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "text": text,
                #"voice": "en_us_male"  # Specify the voice here, e.g., "en_us_male" or "en_uk_female"
            }
        )
        if response.status_code == 200:
            with open(audio_file_path, "wb") as audio_file:
                audio_file.write(response.content)
        else:
            raise Exception(f"Error generating audio: {response.text}")
    
    pygame.mixer.music.load(audio_file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

def record_audio(duration=5, sample_rate=44100):
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_wav_file.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return temp_wav_file.name

def transcribe_audio_v3(audio_file_path):
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    if not DEEPGRAM_API_KEY:
        raise ValueError("Deepgram API key is not set in the environment variables.")

    headers = {
        'Authorization': f'Token {DEEPGRAM_API_KEY}',
    }
    url = 'https://api.deepgram.com/v1/listen'

    with open(audio_file_path, 'rb') as audio_file:
        response = requests.post(
            url,
            headers=headers,
            files={'file': audio_file},
            params={
                'punctuate': 'true',
                'language': 'en'
            }
        )

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to transcribe audio: {response.text}")
