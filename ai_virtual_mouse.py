import cv2
import numpy as np
import time
import mediapipe as mp
import math
import hand_tracking_module as htm
import autopy

# geting the width and height of screen
width_screen, height_screen = autopy.screen.size()

# setting width and height of the frame
width_cam, height_cam = 1080,720

# reducing frames for the mouse trackpad 
frame_redux = 200

# removing jitters
smoothen = 7
prev_loc_x, prev_loc_y = 0, 0
curr_loc_x, curr_loc_y = 0, 0


cap = cv2.VideoCapture(0)
cap.set(3,width_cam)
cap.set(4,height_cam)


prev_time = 0
curr_time = 0


# 1. Find the landmarks
# 2. Get tip of index and middle finger
# 3. Check up fingers
# 4. Only Index -> Moving Mode
# 5. Coordinates transformation
# 6. Reduction of jitters
# 7. Movement of Mouse
# 8. Index and Middle finger up -> Click up
# 9. Distance between these fingers
# 10. Click if short distance

h_detect = htm.hand_detector_tracker(max_hands=1,detection_confidence=0.7)

while True:
    success, img = cap.read()

    img = h_detect.trace_hands(img)
    lm_list,boundary_list = h_detect.get_position(img,draw_bounds=True)
    
    # drawing mouse trackpad
    cv2.rectangle(img,(frame_redux,frame_redux),(width_cam - frame_redux, height_cam - frame_redux),(50,50,50),2)

    if len(lm_list) !=0 :
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]

        fingers = h_detect.finger_up_count()
        # print(fingers)

        # 4 -> 7
        if fingers[1] == 1 and fingers[2] == 0:
            x_transform = np.interp(x1,(frame_redux,width_cam - frame_redux,),(0,width_screen))
            y_transform = np.interp(y1,(frame_redux,height_cam - frame_redux),(0,height_screen))

            curr_loc_x = prev_loc_x + (x_transform - prev_loc_x)/smoothen
            curr_loc_y = prev_loc_y + (y_transform - prev_loc_y)/smoothen

            autopy.mouse.move(width_screen-curr_loc_x, curr_loc_y)

            prev_loc_x, prev_loc_y = curr_loc_x, curr_loc_y

            cv2.circle(img,(x1,y1),10,(100,100,100),cv2.FILLED)

        # 8 -> 10
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, line_info = h_detect.find_distance(8, 12, img)
            if length < 50:
                cv2.circle(img,(line_info[4],line_info[5]),10,(50,50,50),cv2.FILLED)
                autopy.mouse.click()


    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time


    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,3,(255,0,0),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
    
