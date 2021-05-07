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
        '''
            Finds the hands in the given image.

            Parameters : 
                i.  im   : The input image
                ii. draw : Whether the returned image should have landmarks drawn

            Returns : A 2 element tuple
                i.  The first element has the image (same as input if draw=False) 
                ii. The second one has an integer having the number of hands detected 
        '''
        img = im.copy()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        self.res = results.multi_hand_landmarks
        if self.res : 
            for hand in self.res : 
                if draw :
                    self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)
            return (img, len(self.res))
        else : 
            return (img, 0)


    def findPosition(self, im):
        '''
            Finds the position of all the landmarks return them as a list. 
            Refer to :
                https://google.github.io/mediapipe/solutions/hands.html
            for more details on what each index means

            Parameters : 
                i.  im  : The image to analyse 
        '''
        lms = []
        if not self.res:
            self.findHands(im)
        if self.res : 
            for hand in self.res :
                child_ls = []
                for id, lm in enumerate(hand.landmark):
                    h,w,c = im.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    child_ls.append((id, cx, cy))
                lms.append(child_ls)
        return lms


#####     MAIN IMPLEMENTATION     #####
def main() :
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True : 
        success, img = cap.read()
        img, n_hands = detector.findHands(img, True)
        print("Number of hands : ",n_hands)
        ls = detector.findPosition(img)
        if len(ls) > 0:
            print("Length here : ", len(ls[0]))
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__=='__main__':
    main()