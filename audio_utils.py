import os
import pygame # type: ignore
import sounddevice as sd # type: ignore
import wave
import tempfile
from gtts import gTTS # type: ignore
from langdetect import detect, LangDetectException # type: ignore

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
    language = detect_language(text)
    audio_file_path = os.path.join(audio_folder, f"{file_name}.mp3")
    if not os.path.exists(audio_file_path):
        tts = gTTS(text, lang=language)
        tts.save(audio_file_path)
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
    import requests
    from deepgram import Deepgram # type: ignore

    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
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
