import cv2
import mediapipe as mp

class FaceMeshDetector : 

    def __init__(self):
        self.mpH = mp.solutions.holistic
        self.holistic = self.mpH.Holistic()
        self.mpDraw = mp.solutions.drawing_utils
    

    def find(self, im, draw = False):
        '''
            Finds the landmarks in the given image.

            Parameters : 
                i.  im   : The input image
                ii. draw : Whether the returned image should have landmarks drawn

            Returns : 
                i. A single result image ( based on the parameter draw)
        '''
        img = im.copy()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(imgRGB)
        if draw :
            self.mpDraw.draw_landmarks(img, results.face_landmarks, self.mpH.FACE_CONNECTIONS)
            self.mpDraw.draw_landmarks(img, results.left_hand_landmarks, self.mpH.HAND_CONNECTIONS)
            self.mpDraw.draw_landmarks(img, results.right_hand_landmarks, self.mpH.HAND_CONNECTIONS)
            self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpH.POSE_CONNECTIONS)
        return img


#####     MAIN IMPLEMENTATION     #####
def main() :
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector()
    while True : 
        success, img = cap.read()
        img = detector.find(img, True)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__=='__main__':
    main()