import cv2
import time

class CameraHandler:
    def __init__(self, camera_index=0, image_dir='speech_recognition_v2/assets/'):
        """
        Initialize CameraHandler instance.
        :param camera_index: Camera index, default 0 for default camera.
        :param image_dir: Directory to save images, default 'images/'.
        """
        self.cap = cv2.VideoCapture(camera_index)
        self.image_dir = image_dir
        if not self.cap.isOpened():
            raise Exception("Unable to open camera")

    def capture_image(self):
        """
        Capture a frame and save to file with timestamp.
        :return: Path of saved image, returns None if capture fails.
        """
        ret, frame = self.cap.read()
        if ret:
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_path = f"{self.image_dir}captured_image_{timestamp}.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Image saved as {image_path}")
            return image_path
        else:
            print("Unable to capture image")
            return None

    def release(self):
        """Release camera resources."""
        self.cap.release()

# Usage example
if __name__ == "__main__":
    camera_handler = CameraHandler()  # Initialize CameraHandler
    camera_handler.capture_image()  # Capture image and save
    camera_handler.release()  # Release camera resources
