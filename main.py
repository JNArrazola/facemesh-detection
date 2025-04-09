import cv2
import mediapipe as mp
from src.detectors import HolisticDetector

mp_drawing = mp.solutions.drawing_utils

REDUCE_SCALE = 0.5

cap = cv2.VideoCapture(0)
detector = HolisticDetector()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=REDUCE_SCALE, fy=REDUCE_SCALE)
    small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    results = detector.process(small_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            small_frame, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            small_frame, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            small_frame, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            small_frame, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

    output_frame = cv2.resize(small_frame, (frame.shape[1], frame.shape[0]))

    cv2.imshow('MultiDetector Corregido', output_frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

detector.close()
cap.release()
cv2.destroyAllWindows()
