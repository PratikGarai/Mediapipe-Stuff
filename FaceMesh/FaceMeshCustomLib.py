import cv2
import mediapipe as mp

class FaceMeshDetector : 

    def __init__(self, 
        mode=False, 
        maxFaces=5, 
        detection_confidence=0.5,
        track_confidence=0.5 ):

        self.mode = mode
        self.maxFaces = maxFaces
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.mpFm = mp.solutions.face_mesh
        self.facemesh = self.mpFm.FaceMesh(self.mode, self.maxFaces, 
                                        self.detection_confidence, self.track_confidence)
        self.mpDraw = mp.solutions.drawing_utils
    

    def findFaces(self, im, draw = False):
        '''
            Finds the faces in the given image.

            Parameters : 
                i.  im   : The input image
                ii. draw : Whether the returned image should have landmarks drawn

            Returns : A 2 element tuple
                i.  The first element has the image (same as input if draw=False) 
                ii. The second one has an integer having the number of faces detected 
        '''
        img = im.copy()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.facemesh.process(imgRGB)
        self.res = results.multi_face_landmarks
        if self.res : 
            for face in self.res : 
                if draw :
                    self.mpDraw.draw_landmarks(img, face, self.mpFm.FACE_CONNECTIONS)
            return (img, len(self.res))
        else : 
            return (img, 0)


    def findPosition(self, im):
        '''
            Finds the position of all the landmarks return them as a list. 
            Refer to :
                https://google.github.io/mediapipe/solutions/face_mesh.html
            for more details on what each index means

            Parameters : 
                i.  im  : The image to analyse 
        '''
        lms = []
        if not self.res:
            self.findFaces(im)
        if self.res : 
            for face in self.res :
                child_ls = []
                for id, lm in enumerate(face.landmark):
                    h,w,c = im.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    child_ls.append((id, cx, cy))
                lms.append(child_ls)
        return lms


#####     MAIN IMPLEMENTATION     #####
def main() :
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector()
    while True : 
        success, img = cap.read()
        img, n_faces = detector.findFaces(img, True)
        print("Number of faces : ",n_faces)
        ls = detector.findPosition(img)
        if len(ls) > 0:
            print("Length here : ", len(ls[0]))
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__=='__main__':
    main()