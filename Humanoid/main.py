import time

from handler_chat import ChatHandler
from skill_library import side_grasp, lift_up, top_pinch, home_position
from rgmp_framework import initialize_rgmp_framework, process_rgmp_instruction

wake_vocs = ['hello']
end_vocs = ['goodbye', 'bye']

def judge_voc(text):
    for wake_word in wake_vocs:
        if wake_word in text:
            return 1
    for end_word in end_vocs:
        if end_word in text:
            return 2
    return 0


def detect_wake(speech_handler):

    while True:
        audio_path = speech_handler.record_audio()
        text = speech_handler.recognize_speech(audio_path)
        print('Recognition result: '+text)
        if judge_voc(text) == 1:
            speech_handler.play_audio('speech_recognition_v2/assets/di_start.mp3')
            return True
        else:
            time.sleep(0.5)

def perform_action(text, image_path='test.jpg'):
    # Use RGMP framework for action prediction
    try:
        import cv2
        observation = cv2.imread(image_path)
        if observation is not None:
            final_action, pipeline_info = process_rgmp_instruction(text, observation)
            print(f"RGMP predicted action: {final_action}")
            print(f"Selected skill: {pipeline_info['selected_skill']}")
            
            # Execute corresponding skill based on RGMP output
            skill = pipeline_info['selected_skill']
            if skill == 'side_grasp':
                side_grasp(image_path)
            elif skill == 'lift_up':
                lift_up(image_path)
            elif skill == 'top_pinch':
                top_pinch(image_path)
            
            home_position()
        else:
            print(f"Could not load image: {image_path}")
    except Exception as e:
        print(f"RGMP processing failed: {e}")
        # Fallback to original logic
        if 'side_grasp' in text:
            side_grasp(image_path)
            home_position()
        elif 'lift_up' in text:
            lift_up(image_path)
            home_position()
        elif 'top_pinch' in text:
            top_pinch(image_path)
            home_position()
        

def main():
    chat_handler = ChatHandler()

    begin_chat = False
    while True:
        if begin_chat is False:
            if detect_wake(speech_handler=chat_handler.speech_handler):
                text = 'Hello'
                begin_chat = True
            else:
                pass
        else:
            text = chat_handler.recognize_audio()
            if judge_voc(text) == 2:
                text = 'Goodbye'
                begin_chat = False
                chat_handler.chat(text, use_history=False)
        
        if begin_chat:
            response = chat_handler.chat(text, use_history=False)
            print('Complete output: '+response)
            # Get latest captured image for skill execution
            latest_image = 'speech_recognition_v2/assets/latest_capture.jpg'
            perform_action(response, latest_image)

if __name__ == '__main__':
    main()