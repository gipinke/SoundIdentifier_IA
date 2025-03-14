import wave
import pyaudio
import os

from constants import constants

audio = pyaudio.PyAudio()


def create_audio_file(audio_data, filename):
    waveFile = wave.open(filename, 'wb')
    waveFile.setnchannels(constants.CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(constants.FORMAT))
    waveFile.setframerate(constants.RATE)
    waveFile.writeframes(b''.join(audio_data))
    waveFile.close()


def delete_audio_file(filename):
    os.remove(filename)
