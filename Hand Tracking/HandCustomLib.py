import cv2
import mediapipe as mp

class HandDetector : 

    def __init__(self, 
        mode=False, 
        maxHands=2, 
        detection_confidence=0.5,
        track_confidence=0.5 ):

        self.mode = mode
        self.maxHands = maxHands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 
                                        self.detection_confidence, self.track_confidence)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, im, draw = False):
        img = im.copy()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        
        if results.multi_hand_landmarks : 
            for hand in results.multi_hand_landmarks : 
                if draw :
                    self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)
        return img



def main() :
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True : 
        success, img = cap.read()
        img = detector.findHands(img, True)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__=='__main__':
    main()
