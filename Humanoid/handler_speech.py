import pyaudio
import wave
import json
import base64
import time
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import urlencode, quote_plus
import sys
import yaml
import requests
import os
import pygame
import random
import time
import numpy as np


class SpeechHandler:
    def __init__(self, config_file='speech_recognition_v2/configs.yaml', cuid="123456PYTHON", dev_pid=80001, rate=16000): 
        configs = yaml.safe_load(open(config_file))
        self.API_KEY = configs['baidu']['api_key']
        self.SECRET_KEY = configs['baidu']['secret_key']
        self.CUID = cuid
        self.DEV_PID = dev_pid
        self.RATE = rate

        self.ASR_URL = 'http://vop.baidu.com/pro_api'
        self.TOKEN_URL = 'http://aip.baidubce.com/oauth/2.0/token'
        self.TTS_URL = 'http://tsn.baidu.com/text2audio'
        self.token = self.fetch_token()

        self.asset_dir = 'speech_recognition_v2/assets/'

        # Silence threshold, smaller value means more sensitive to sound
        self.silence_threshold = 1000
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.RECORD_SECONDS = 2
        self.audio = pyaudio.PyAudio()

    # Get access_token
    def fetch_token(self):
        params = {
            'grant_type': 'client_credentials',
            'client_id': self.API_KEY,
            'client_secret': self.SECRET_KEY
        }
        response = requests.post(self.TOKEN_URL, data=params).json()
        if 'access_token' in response:
            return response['access_token']
        else:
            raise Exception('Unable to fetch access token')
        
    # Check if there is currently sound
    def is_speech(self):
        # Record for fixed time, check if greater than set threshold, return True or False
        stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)

        frames = []
        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        # audio.terminate()

        audio_data = b''.join(frames)
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        max_volume = np.max(np.abs(audio_data))
        print(f"Max volume: {max_volume}")

        return max_volume > self.silence_threshold
    
    # Record for fixed time and save
    def record_audio(self):
        stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)

        frames = []
        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        # audio.terminate()

        audio_file = 'speech_recognition_v2/assets/real_audio.wav'
        with wave.open(audio_file, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))

        return audio_file


    # Real-time audio recognition and return result
    def recognize_audio(self):
        # audio = pyaudio.PyAudio()
        stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)

        print("Listening for speech...")

        frames = []
        silence_frames = 0
        silence_threshold = self.silence_threshold  
        max_silence = int(self.RATE / self.CHUNK * 3)  

        while True:
            data = stream.read(self.CHUNK)
            frames.append(data)
            volume = max(data)

            # Detect silence
            if volume < silence_threshold:
                silence_frames += 1
            else:
                silence_frames = 0

            # Stop recording if silence exceeds specified time
            if silence_frames > max_silence:
                print("Detected silence. Stopping recording.")
                break

        stream.stop_stream()
        stream.close()
        # audio.terminate()

        # Write audio to PCM file
        audio_file = 'speech_recognition_v2/assets/realtime_audio.pcm'
        with open(audio_file, 'wb') as f:
            f.write(b''.join(frames))

        # Perform speech recognition and return result
        result = self.recognize_speech(audio_file)
        return result

    # Speech recognition
    def recognize_speech(self, audio_file='speech_recognition_v2/assets/realtime_audio.pcm'):
        with open(audio_file, 'rb') as speech_file:
            speech_data = speech_file.read()

        length = len(speech_data)
        speech_base64 = base64.b64encode(speech_data).decode('utf-8')
        params = {
            'dev_pid': self.DEV_PID,
            'format': 'pcm',
            'rate': self.RATE,
            'token': self.token,
            'cuid': self.CUID,
            'channel': 1,
            'speech': speech_base64,
            'len': length
        }

        post_data = json.dumps(params).encode('utf-8')
        req = Request(self.ASR_URL, post_data)
        req.add_header('Content-Type', 'application/json')

        try:
            start_time = time.time()
            response = urlopen(req)
            result_str = response.read().decode('utf-8')
            print(f"Request time: {time.time() - start_time} seconds")
            result_json = json.loads(result_str)
            if "result" in result_json:
                recognition_result = result_json["result"][0]
                
                print(f"Recognized: {recognition_result}")
                return recognition_result
            else:
                return None
        except URLError as err:
            print(f"ASR request failed: {err.code}")
            result_str = err.read().decode('utf-8')
            return None
        
        

    # Play audio file
    def play_audio(self, audio_file):
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        # If audio is playing, wait for playback to complete
        while pygame.mixer.music.get_busy():  
            time.sleep(0.2)

        # Ensure audio stops and releases resources after playback
        pygame.mixer.music.stop()
        pygame.mixer.quit()


    # Generate audio from corresponding text
    def generate_audio(self, text):
        # Speaker configuration, speed, pitch, volume, etc.
        PER = 106  # Speaker
        SPD = 7    # Speed
        PIT = 4    # Pitch
        VOL = 15    # Volume
        AUE = 3    # File format mp3

        # Get Token
        token = self.token
        params = {
            'tok': token,
            'tex': quote_plus(text),
            'per': PER,
            'spd': SPD,
            'pit': PIT,
            'vol': VOL,
            'aue': AUE,
            'cuid': self.CUID,
            'lan': 'zh',
            'ctp': 1
        }
        response = requests.post(self.TTS_URL, params=params)
        content_type = response.headers['Content-Type']

        # Save audio file
        if 'audio/' in content_type:    
            # Generate filename based on current timestamp
            filename = f"{self.asset_dir}audio_{int(time.time() * 1000)}.mp3"
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Audio saved as {filename}")
        else:
            print("Error:", response.text)
            return None
        
        return filename
    
    # Generate audio and play
    def generate_play_audio(self, text):
        audio_file = self.generate_audio(text)
        if audio_file:
            self.play_audio(audio_file)
        else:
            print("Failed to generate audio file.")


# Main function
if __name__ == '__main__':
    speech_handler = SpeechHandler()

    # Example text
    text = "Hello, I am a humanoid robot. How can I help you?"
    result = speech_handler.generate_play_audio(text)

    result_str = speech_handler.recognize_audio()
    print(result_str)

    result_str = 'You just said: ' + result_str
    speech_handler.generate_play_audio(result_str)

    while True:
        if speech_handler.is_speech():
            print("Detected speech.")
        else:
            print("No speech detected.")