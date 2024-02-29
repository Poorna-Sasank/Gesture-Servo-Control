import mediapipe as mp
import cv2
import numpy as np
import time
import math

from pyfirmata import Arduino, SERVO
from time import sleep

PORT = ''
board = Arduino(PORT)
PIN = board.get_pin('d:6:s')

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
def move_servo(angle):
    PIN.write(angle)
    
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print('Ignoring empty camera frame')
            continue
        
        lml = []
        xl = []
        yl = []
        
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = hands.process(img)
        
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
            for id, lm in enumerate(results.multi_hand_landmarks[0].landmark):
                h, w, _ = img.shape
                xc, yc = int(lm.x * w), int(lm.y * h)
                lml.append([id, xc, yc])
                xl.append(xc)
                yl.append(yc)
            x1, y1 = lml[4][1], lml[4][2]
            x2, y2 = lml[8][1], lml[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img, (x1, y1), 10, (255, 0, 128), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 128), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 128), 3)
        # cv2.circle(image, (cx, cy), 10, (255, 0, 128), cv2.FILLED)
            distance = math.hypot(x2 - x1, y2 - y1)
            move_servo(distance)
            cv2.putText(img, str(int(distance)), (cx+30, cy), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 128), 3)
        else:
            move_servo(90)
        cv2.imshow('Hands', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()

