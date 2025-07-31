import threading
import queue
import time
import os

from prompts import *
from handler_api import QwenVLConversation
from handler_speech import SpeechHandler
from handler_camera import CameraHandler
from gss_framework import initialize_gss_framework, process_gss_instruction


class ChatHandler:
    def __init__(self):
        self.conversation = QwenVLConversation()
        self.speech_handler = SpeechHandler()
        self.camera_handler = CameraHandler()
        self.gss_framework = initialize_gss_framework()

        self.asset_dir = 'speech_recognition/assets/'
        self.END_AUDIO = 'di_end.mp3'
        self.START_AUDIO = 'di_strart.mp3'

        self.threshold = 0.5
        self.cur_response = ''

    def clean_format(self, res):
        extra_chars = ['side_grasp', 'lift_up', 'top_pinch', '()']
        for s in extra_chars:
            if s in res:
                res = res.replace(s, '')
        return res


    # Use two threads for streaming audio generation and sequential playback
    def chat(self, query, use_image=True, use_history=True):
        if use_image:
            user_image = self.camera_handler.capture_image()
        else:
            user_image = None

        # Process through GSS framework for skill selection
        if user_image and self._is_manipulation_request(query):
            gss_result = process_gss_instruction(user_image, query)
            skill_info = f"Selected skill: {gss_result['selected_skill']}, Shape: {gss_result['shape_info']}"
            user_input = MAIN_PROMPT.format(memorys=MEMORYS, questions=f"{query}\nGSS Analysis: {skill_info}")
        else:
            user_input = MAIN_PROMPT.format(memorys=MEMORYS, questions=query)
            
        self.cur_response = ''
        response_stream = self.conversation.interact(user_input=user_input, user_image=user_image, use_history=use_history)

        def generate_audio():
            for sentence in response_stream:
                print(f"Generating audio for sentence: {sentence}")
                self.cur_response += sentence
                if sentence == "END":
                    break
                audio_file = self.speech_handler.generate_audio(self.clean_format(sentence))
                print(f"Generated audio file: {audio_file}")
                audio_queue.put(audio_file)

        def play_audio():
            while True:
                audio_file = audio_queue.get()
                print(f"Playing audio file: {audio_file}")
                if audio_file == "END":
                    # Play end sound
                    self.speech_handler.play_audio(os.path.join(self.asset_dir, self.END_AUDIO))
                    break
                # Play speech
                self.speech_handler.play_audio(audio_file)
                # Delete after playback
                os.remove(audio_file)

        audio_queue = queue.Queue()
        audio_thread = threading.Thread(target=play_audio)
        audio_thread.start()

        generate_audio()
        audio_queue.put("END")
        audio_thread.join()

        # Delete image and return complete string after completion
        if 'END' in self.cur_response:
            if user_image and os.path.exists(user_image):
                os.remove(user_image)
            return self.cur_response.replace('END', '')

    def recognize_audio(self):
        return self.speech_handler.recognize_audio()
    
    def _is_manipulation_request(self, query):
        """Check if the query is a manipulation request"""
        manipulation_keywords = ['grasp', 'pick', 'lift', 'pinch', 'grab', 'take', 'get']
        return any(keyword in query.lower() for keyword in manipulation_keywords)
        
    


def main():
    chat_handler = ChatHandler()

    while True:
        # TODO Detect start and end

        text = chat_handler.recognize_audio()
        chat_handler.chat(text, use_history=False)

    

if __name__ == '__main__':

    main()