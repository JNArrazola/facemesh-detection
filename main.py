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

from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
import sys

mp_drawing = mp.solutions.drawing_utils
REDUCE_SCALE = 0.5  # Reduce the scale of the image to speed up processing

class MultiDetectorApp(QWidget):
    def __init__(self):
        """  
        Initializes the MultiDetectorApp class.
        Sets up the GUI with a QLabel for displaying video frames and a QPushButton to start/stop detection.
        The class uses OpenCV for video capture and MediaPipe for landmark detection.
        The class contains methods to toggle detection, update frames, and handle GUI events.
        """

        super().__init__() # Initialize the QWidget class
        self.setWindowTitle("MultiDetector GUI") # Set the window title
        self.setGeometry(100, 100, 800, 600) # Set the window size

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.button = QPushButton("Iniciar / Detener", self) # Create a button to start/stop detection
        self.button.clicked.connect(self.toggle_detection)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        self.setLayout(layout)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.detector = HolisticDetector()

        self.running = False

    def toggle_detection(self):
        """
        Toggles the detection on and off.
        If detection is off, it starts the video capture and timer.
        If detection is on, it stops the video capture and timer.
        """

        if not self.running: # If detection is off
            self.cap = cv2.VideoCapture(0) # Open the webcam
            self.running = True # Set running to True
            self.timer.start(30) # Start the timer to update frames
        else: # If detection is on
            self.running = False # Set running to False
            self.timer.stop()  # Stop the timer
            if self.cap: # If the video capture is open
                self.cap.release() # Release the webcam
            self.label.clear() # Clear the label

    def update_frame(self):
        """  
        Captures a frame from the webcam, processes it using the HolisticDetector,
        and displays the processed frame in the QLabel.
        The method resizes the frame to reduce processing time and converts it to RGB format.
        It draws landmarks for the face, hands, and body on the frame.
        The processed frame is then converted to QImage and displayed in the QLabel.
        The method also handles the case where the video capture is not opened or the frame capture fails.
        If the video capture is not opened, it returns without doing anything.
        If the frame capture fails, it returns without doing anything.
        The method uses OpenCV for video capture and display, and MediaPipe for landmark detection.
        """
 
        # Capture a frame from the webcam
        if not self.cap or not self.cap.isOpened():
            return

        success, frame = self.cap.read()
        if not success: # If frame capture fails
            return # Return without doing anything

        small_frame = cv2.resize(frame, (0, 0), fx=REDUCE_SCALE, fy=REDUCE_SCALE)  # Resize the frame
        small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB

        results = self.detector.process(small_rgb)  # Process the frame using the HolisticDetector

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

        output_frame = cv2.resize(small_frame, (frame.shape[1], frame.shape[0]))  # Resize the frame back to original size

        # Convert to QImage and show in QLabel
        img = QImage(output_frame.data, output_frame.shape[1], output_frame.shape[0],
                     output_frame.strides[0], QImage.Format.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(img))

    def closeEvent(self, event):
        """  
        Handles the close event of the application.
        Stops the timer, releases the video capture, and closes the detector.
        """
        
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.detector.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiDetectorApp()
    window.show()
    sys.exit(app.exec())
