import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Dimensions of Input
HEIGHT = 480
WIDTH = 640
CHANNELS = 3

while True : 
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks : 
        for hand in results.multi_hand_landmarks : 
            for id, lm in enumerate(hand.landmark):
                cx = int(lm.x*WIDTH)
                cy = int(lm.y*HEIGHT)
            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

    cv2.putText(img, "FPS : "+str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)