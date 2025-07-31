import sounddevice as sd
import soundfile as sf
import numpy as np
import webrtcvad
import os
import time
import sounddevice as sd
import numpy as np
import pygame


class RealTimeAudioRecorder:
    def __init__(self, silence_threshold=3000, sample_rate=16000, chunk_duration=0.5, vad_mode=3, start_trigger_count=2, stop_trigger_count=3):
        self.silence_threshold = silence_threshold  
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration  
        self.start_trigger_count = start_trigger_count  
        self.stop_trigger_count = stop_trigger_count  

  
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(vad_mode)  

    def is_speech(self, audio_chunk):
        """ Use VAD and volume threshold to determine if it's valid human speech """
        # Split into 30ms frames to reduce false positives
        audio_chunk = audio_chunk.flatten().tobytes()  # Convert to bytes
        frame_duration = 0.03  # 30ms
        frame_length = int(self.sample_rate * frame_duration * 2)  # 16-bit PCM so multiply by 2
        
        for i in range(0, len(audio_chunk), frame_length):
            frame = audio_chunk[i:i + frame_length]
            if len(frame) < frame_length:
                continue
            # VAD detection
            if self.vad.is_speech(frame, self.sample_rate) and np.max(np.abs(np.frombuffer(frame, dtype=np.int16))) > self.silence_threshold:
                return True
        return False

    # Play audio file at specified path
    def play_audio_file(self, audio_file):
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.5)
        pygame.mixer.music.stop()
        pygame.mixer.quit()


    def record_audio(self):
        while True:
            print("Starting to listen...")
            frames = []
            recording_started = False
            speech_detected_count = 0  # Speech detection count
            silence_detected_count = 0  # Silence detection count

            while True:
                # Record chunk_duration seconds of audio data
                chunk = sd.rec(int(self.sample_rate * self.chunk_duration), samplerate=self.sample_rate, channels=1, dtype='int16')
                sd.wait()

                # Detect if it's human speech
                if self.is_speech(chunk):
                    speech_detected_count += 1
                    silence_detected_count = 0  # Reset silence detection count

                    if not recording_started:
                        print(f"Detected speech: {speech_detected_count} times")

                    # If enough consecutive speech blocks detected, start recording
                    if speech_detected_count >= self.start_trigger_count and not recording_started:
                        frames.append(chunk)
                        recording_started = True
                        print("Starting recording...")
                    elif recording_started:
                        frames.append(chunk)
                        print(f"Recording...")
                else:
                    silence_detected_count += 1
                    speech_detected_count = 0  # Reset speech detection count
                    print(f"Detected silence: {silence_detected_count} times")

                    # If enough consecutive silence blocks detected, end recording
                    if silence_detected_count >= self.stop_trigger_count and recording_started:
                        print("Detected silence, ending recording...")
                        break

            # Concatenate audio data into complete recording
            if recording_started and frames:
                audio_data = np.concatenate(frames, axis=0)

                # Save audio file with fixed filename
                sf.write(self.record_dir, audio_data, self.sample_rate)
                print(f"Recording saved to {self.record_dir}")

                # Upload photo
                self.capture_handler.capture()
                self.syn_handler.upload_file("snapshot.jpg")
                # Upload recording file
                self.syn_handler.upload_file("recording.wav")

                # Delete existing audio.mp3
                print("Waiting for model output playback...")
                # Download model output
                if self.syn_handler.fetch_file("audio.mp3"):
                    self.play_audio_file(self.play_dir)
                    os.remove(self.play_dir)
                    self.syn_handler.delete_file("audio.mp3")


if __name__ == "__main__":

    recorder = RealTimeAudioRecorder()
    while True:
        recorder.record_audio()

  