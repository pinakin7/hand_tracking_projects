import cv2
import time
import numpy as np
import hand_tracking_module as htm 
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


# setting height and width of the frame
width_cam, height_cam = 1080,720

cap = cv2.VideoCapture(0)
cap.set(3,width_cam)
cap.set(4,height_cam)


prev_time = 0
curr_time = 0

# initializing the hand_detector_tracker object with detection_confidence as 0.8 in order to avoid detecting jitters
h_detect = htm.hand_detector_tracker(detection_confidence=0.8)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volume_range = volume.GetVolumeRange()

min_volume = volume_range[0]
max_volume = volume_range[1]


vol_bar = 400
vol_percent = 100

while True:
    success, img = cap.read()

    img = h_detect.trace_hands(img)
    landmarks_list, boundaries_list = h_detect.get_position(img,draw=False)
    if len(landmarks_list) != 0:
        # we'd be needing id_ number 4 and id_ number 8
        # print(landmarks_list)
        x1, y1, x2, y2 = landmarks_list[4][1], landmarks_list[4][2], landmarks_list[8][1], landmarks_list[8][2]
        center_x, center_y = (x1+x2)//2, (y1+y2)//2
        cv2.circle(img,(x1,y1),10,(0,255,0), cv2.FILLED)
        cv2.circle(img,(x2,y2),10,(0,255,0), cv2.FILLED)
        cv2.circle(img,(center_x,center_y),10,(255,255,0), cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0), 2)


        length = math.hypot(x2-x1, y2-y1)

        # hand range = 50 -> 300
        # volume range = - 65 -> 0

        vol = np.interp(length,[50,200], [min_volume, max_volume])
        vol_bar = np.interp(length,[50,200], [400, 150])
        vol_percent = np.interp(length,[50,200], [0, 100])

        volume.SetMasterVolumeLevel(vol,None)

        if length < 50:
            cv2.circle(img,(center_x,center_y),10,(0,0,255), cv2.FILLED)

        
    cv2.rectangle(img,(50,150),(80,400),(0,0,255), 2)
    cv2.rectangle(img,(50,int(vol_bar)),(80,400),(0,0,255), cv2.FILLED)

    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img,str(int(fps)),(10,50),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(255,0,0),3)
    cv2.putText(img,f' {int(vol_percent)} %',(50,100),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(255,0,0),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)