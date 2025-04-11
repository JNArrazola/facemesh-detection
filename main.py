"""  
Main script to run the MultiDetector application.

This script captures video from the webcam and processes it using the HolisticDetector class.
It detects and draws landmarks for the face, hands, and body in real-time.
The script uses OpenCV for video capture and display, and MediaPipe for landmark detection.
"""

# Import necessary libraries
import cv2
import mediapipe as mp
from src.detectors import HolisticDetector

mp_drawing = mp.solutions.drawing_utils

REDUCE_SCALE = 0.5 # Reduce the scale of the image to speed up processing

cap = cv2.VideoCapture(0) 
detector = HolisticDetector() 

# While the video capture is opened, read frames from the webcam
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=REDUCE_SCALE, fy=REDUCE_SCALE) # Resize the frame
    small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB) # Convert the frame to RGB

    results = detector.process(small_rgb) # Process the frame using the HolisticDetector

    # Draw landmarks on the frame
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

    output_frame = cv2.resize(small_frame, (frame.shape[1], frame.shape[0])) # Resize the frame back to original size

    cv2.imshow('MultiDetector', output_frame) # Display the frame
    if cv2.waitKey(5) & 0xFF == 27:
        break

detector.close()
cap.release()
cv2.destroyAllWindows()
