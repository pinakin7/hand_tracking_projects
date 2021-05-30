import cv2 
import mediapipe as mp 
import time
import math

# creating a class in order to use this as a module
class hand_detector_tracker():
    # defining constructor for the class to parse the parameters required in the Hands instance of the mediapipe module
    def __init__(self,static_mode = False, max_hands = 2,detection_confidence = 0.5,tracking_confidence = 0.5):
        self.mode = static_mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode,self.max_hands,self.detection_confidence,self.tracking_confidence)

        self.mp_draw = mp.solutions.drawing_utils

        self.tip_id = [4, 8, 12, 16, 20]
        

    # function to trace the hands and draw lines on the palm if needed
    def trace_hands(self,img,draw = True):

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, handlms, self.mp_hands.HAND_CONNECTIONS)

        return img

    # fuction to get the exact landmarks of the points plotted on the hand
    def get_position(self,img,hand_no=0,draw=True,draw_bounds=False):
        self.boundary_box = []
        x_list = []
        y_list = []
        self.lm_list = []

        if self.results.multi_hand_landmarks:
            handlms = self.results.multi_hand_landmarks[hand_no]
            for id_,lm in enumerate(handlms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                x_list.append(cx)
                y_list.append(cy)
                self.lm_list.append([id_, cx, cy])
                if draw:
                    cv2.circle(img,(cx,cy),10,(0,255,255), cv2.FILLED)

            x_min, x_max, y_min, y_max = min(x_list), max(x_list), min(y_list), max(y_list)
            if draw_bounds:
                cv2.rectangle(img,(x_min,y_min),(x_max,y_max),(0,255,0),2)
            self.boundary_box = x_min, x_max, y_min, y_max

        return self.lm_list, self.boundary_box

    def finger_up_count(self):
        fingers = []
        if self.lm_list[self.tip_id[0]][1] > self.lm_list[self.tip_id[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        for id_ in range(1,5):
            if self.lm_list[self.tip_id[id_]][2] < self.lm_list[self.tip_id[id_] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers


    def find_distance(self,p1,p2,img,draw=True,r=10,t=2):
        x1, y1, x2, y2 = self.lm_list[p1][1], self.lm_list[p1][2], self.lm_list[p2][1], self.lm_list[p2][2]
        center_x, center_y = (x1+x2)//2, (y1+y2)//2
        
        if draw:
            cv2.circle(img,(x1,y1),r,(200,200,200), cv2.FILLED)
            cv2.circle(img,(x2,y2),r,(200,200,200), cv2.FILLED)
            cv2.circle(img,(center_x,center_y),r,(10,10,10), cv2.FILLED)
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0), t)


        self.length = math.hypot(x2-x1, y2-y1)

        return self.length, img, [x1,y1,x2,y2,center_x,center_y]


def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    curr_time = 0
    hdt = hand_detector_tracker()
    while True:
        success, img = cap.read()

        img = hdt.trace_hands(img)
        lm_list, boundary_box = hdt.get_position(img)

        if len(lm_list) != 0:
            print(lm_list[5])
            print(boundary_box)

        curr_time = time.time()
        fps = 1/(curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,3,(255,0,0),3)
        cv2.imshow("Image",img)
        cv2.waitKey(1)
        

if __name__ == '__main__':
    main()