import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

class HolisticDetector:
    def __init__(self, model_complexity=1):
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
