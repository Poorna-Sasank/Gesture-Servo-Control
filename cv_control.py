import cv2
import numpy as np
import math
from time import sleep
from pyfirmata import Arduino, SERVO
import mediapipe as mp

class ServoController:
    def __init__(self, port, pin):
        self.board = Arduino(port)
        self.pin = self.board.get_pin(f'd:{pin}:s')

    def move_servo(self, angle):
        self.pin.write(angle)

class HandDetector:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def detect_hands(self, img):
        img_rgb = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        return results

    def draw_landmarks(self, img, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

class HandServoApp:
    def __init__(self, servo_controller):
        self.servo_controller = servo_controller
        self.hand_detector = HandDetector()
        self.cap = cv2.VideoCapture(0)

    def run(self):
        while self.cap.isOpened():
            ret, img = self.cap.read()
            if not ret:
                print('Ignoring empty camera frame')
                continue
            
            results = self.hand_detector.detect_hands(img)
            self.hand_detector.draw_landmarks(img, results)

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                x1, y1 = int(landmarks[4].x * img.shape[1]), int(landmarks[4].y * img.shape[0])
                x2, y2 = int(landmarks[8].x * img.shape[1]), int(landmarks[8].y * img.shape[0])
                
                distance = math.hypot(x2 - x1, y2 - y1)
                self.servo_controller.move_servo(distance)
                
                cv2.circle(img, (x1, y1), 10, (255, 0, 128), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 128), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 128), 3)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.putText(img, str(int(distance)), (cx + 30, cy), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 128), 3)
            else:
                self.servo_controller.move_servo(90)  # Default position when no hands are detected
            
            cv2.imshow('Hands', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    PORT = ''  # Arduino com port
    PIN = 6    
    servo_controller = ServoController(PORT, PIN)
    app = HandServoApp(servo_controller)
    app.run()
