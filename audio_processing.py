# audio_processing.py

import wave
import streamlit as st
import pyaudio  # type: ignore
import threading
from pydub import AudioSegment  # type: ignore

# Constants for audio recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 16kHz sample rate
CHUNK = 1024
OUTPUT_WAV = "output.wav"
OUTPUT_MP3 = "output.mp3"

class AudioRecorder:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.recording = False
        self.record_thread = None

    def start_recording(self):
        if self.recording:
            st.warning("Recording is already in progress.")
            return
        try:
            self.stream = self.p.open(format=FORMAT,
                                        channels=CHANNELS,
                                        rate=RATE,
                                        input=True,
                                        frames_per_buffer=CHUNK)
        except Exception as e:
            st.error(f"Could not open microphone: {str(e)}")
            st.session_state["debug_log"].append(f"Microphone open error: {str(e)}")
            return

        self.recording = True
        self.frames = []
        st.info("Recording started...")
        st.session_state["debug_log"].append("Recording started.")

        def record():
            frame_count = 0
            while self.recording:
                try:
                    data = self.stream.read(CHUNK)
                    self.frames.append(data)
                    frame_count += 1
                    # Log every 100 frames to avoid log overflow
                    if frame_count % 100 == 0:
                        st.session_state["debug_log"].append(f"Captured frame {frame_count}")
                except Exception as e:
                    self.recording = False
                    st.error(f"Recording error: {str(e)}")
                    st.session_state["debug_log"].append(f"Recording exception: {str(e)}")
                    break

        self.record_thread = threading.Thread(target=record, daemon=True)
        self.record_thread.start()

    def stop_recording(self):
        if not self.recording:
            st.warning("No recording in progress to stop.")
            return None
        self.recording = False
        self.record_thread.join()

        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

        st.session_state["debug_log"].append("Recording stopped.")

        if not self.frames:
            st.error("No audio data was captured.")
            st.session_state["debug_log"].append("No frames captured during recording.")
            return None

        # Save WAV file
        try:
            wf = wave.open(OUTPUT_WAV, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            st.success(f"Recording stopped. Audio saved to {OUTPUT_WAV}")
            st.session_state["debug_log"].append(f"Saved WAV file: {OUTPUT_WAV}")
        except Exception as e:
            st.error(f"Error saving WAV file: {str(e)}")
            st.session_state["debug_log"].append(f"WAV save error: {str(e)}")
            return None

        # Convert WAV to MP3 using pydub
        try:
            audio = AudioSegment.from_wav(OUTPUT_WAV)
            audio.export(OUTPUT_MP3, format="mp3")
            st.success(f"Audio converted to MP3 and saved to {OUTPUT_MP3}")
            st.session_state["debug_log"].append(f"Converted to MP3: {OUTPUT_MP3}")
            return OUTPUT_MP3
        except Exception as e:
            st.error(f"Error converting WAV to MP3: {str(e)}")
            st.session_state["debug_log"].append(f"MP3 conversion error: {str(e)}")
            return None

def get_recorder():
    """
    Retrieves the singleton AudioRecorder instance from session state.
    If it doesn't exist, creates a new instance and stores it.
    """
    if 'recorder' not in st.session_state:
        st.session_state['recorder'] = AudioRecorder()
    return st.session_state['recorder']
