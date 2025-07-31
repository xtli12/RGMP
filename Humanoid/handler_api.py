import os
from dashscope import MultiModalConversation
import re
import time
import yaml

class QwenVLConversation:
    def __init__(self, config_file='speech_recognition_v2/configs.yaml'):
        self.configs = yaml.safe_load(open(config_file))
        self.MODEL_NAME = self.configs['qwen']['model_name']
        self.API_KEY = self.configs['qwen']['api_key']
        self.messages = []

    def add_message(self, role, content):
        """Adds a message to the conversation history."""
        self.messages.append({"role": role, "content": content})

    def generate_response_stream(self):
        """
        Calls the API to generate a response in streaming mode.
        
        Yields one sentence at a time based on punctuation marks (e.g., periods, commas).
        """
        responses = MultiModalConversation.call(
            api_key=self.API_KEY,
            model=self.MODEL_NAME,
            messages=self.messages,
            stream=True,
            incremental_output=True
        )

        buffer = ""  # Temporary storage for accumulating partial sentences
        for response in responses:
            try:
                if (
                    "output" in response and
                    "choices" in response["output"] and
                    response["output"]["choices"] and
                    "message" in response["output"]["choices"][0] and
                    "content" in response["output"]["choices"][0]["message"] and
                    response["output"]["choices"][0]["message"]["content"]
                ):
                    text_output = response["output"]["choices"][0]["message"]["content"][0]["text"]
                    buffer += text_output  # Accumulate text
                    # Split buffer into complete sentences
                    sentences = re.split(r'([.!?])', buffer)  # Keep punctuation
                    for i in range(0, len(sentences) - 1, 2):
                        yield sentences[i] + sentences[i + 1]  # Yield complete sentences
                    buffer = sentences[-1]  # Keep the remaining partial sentence
            except Exception as e:
                print(f"Error: {e}")

        # Yield remaining buffer content if not empty
        if buffer.strip():
            yield buffer.strip()

        # Yield the END marker to signify completion
        yield "END"

    def interact(self, user_input, user_image=None, use_history=True):
        """
        Main method to interact with the model. It supports multi-round conversations.
        
        :param user_input: Text input from the user.
        :param user_image: URL of an image input from the user.
        :param clear: Whether to clear the conversation history.
        :return: Stream generator for the response.
        """
        # Add user input to messages
        content = []
        if user_image:
            content.append({"image": user_image})
        if user_input:
            content.append({"text": user_input})
        if use_history is False:
            self.messages = []
        self.add_message("user", content)
        
        # Return the response stream generator
        return self.generate_response_stream()


# Usage example
if __name__ == "__main__":
    conversation = QwenVLConversation()

    # First round of conversation: input image and question
    print("First round conversation content:")
    response_stream = conversation.interact(
        user_input="What is this?",
        user_image="speech_recognition_v2/assets/example1.jpg"
    )

    # Loop to receive and print sentences until END marker
    for sentence in response_stream:
        if sentence == "END":
            break
        print(sentence)
        # time.sleep(0.5)  

    # Second round of conversation: input new question
    print("\nSecond round conversation content:")
    response_stream = conversation.interact(
        user_input="Write a poem describing this scene"
    )

    for sentence in response_stream:
        if sentence == "END":
            break
        print(sentence)
        # time.sleep(0.5)  
