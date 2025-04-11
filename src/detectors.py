"""  
detectors.py

This module contains the implementation of the HolisticDetector class.
The HolisticDetector class uses MediaPipe's holistic model to detect and draw landmarks for the face, hands, and body in real-time.
The class provides methods to initialize the detector, process images, and close the detector.
The HolisticDetector class is initialized with a model complexity parameter that determines the complexity of the model used for detection.
"""

# Import necessary libraries
import mediapipe as mp

# Import MediaPipe solutions for holistic detection
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define the HolisticDetector class
class HolisticDetector:
    def __init__(self, model_complexity=1):
        """  
        Parameters:
        * model_complexity (int): Complexity of the model. 0: Lite, 1: Full. Default is 1.
        * static_image_mode (bool): Whether to treat input images as a batch of static images or a video stream. Default is False.
        * smooth_landmarks (bool): Whether to smooth landmarks across frames. Default is True.
        * enable_segmentation (bool): Whether to enable segmentation. Default is False.
        * refine_face_landmarks (bool): Whether to refine face landmarks. Default is True.
        * min_detection_confidence (float): Minimum confidence value for detection. Default is 0.5.
        * min_tracking_confidence (float): Minimum confidence value for tracking. Default is 0.5.
        """
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, image):
        return self.holistic.process(image)

    def close(self):
        self.holistic.close()
