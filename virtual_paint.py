import os
import time
import mediapipe as mp
import cv2
import numpy as np
import hand_tracking_module as htm 

ui_path = "Virtual Paint UI"

ui_list = os.listdir(ui_path)

ui_overlay_dict = dict()

for img_path in ui_list:
    image = cv2.imread(os.path.join(ui_path, img_path))
    ui_overlay_dict[img_path] = image


# print(ui_overlay_dict)

width_cam, height_cam = 1280,720

cap = cv2.VideoCapture(0)
cap.set(3,width_cam)
cap.set(4,height_cam)

prev_time = 0
curr_time = 0

h_detect = htm.hand_detector_tracker(max_hands=1,detection_confidence=0.85)

header = ui_overlay_dict['paint_ui.png']
draw_color = (255,255,255)

brush_thickness = 10
eraser_thickness = 25

x_prev,y_prev = 0,0

paint_canvas = np.zeros((height_cam,width_cam,3),np.uint8)

while True:
    success, img = cap.read()


    # flipping the image
    img = cv2.flip(img,1)


    # detecting hand landmarks
    img = h_detect.trace_hands(img)
    lm_list,boundary_list = h_detect.get_position(img,draw_bounds=False)

    if len(lm_list) != 0:
        # print(lm_list)
    
        # tip of index finger
        x_index,y_index = lm_list[8][1:]

        # tip of middle finger
        x_middle,y_middle = lm_list[12][1:]

        # checking if the tips of fingers are up or not
        fingers = h_detect.finger_up_count()
        # print(fingers)

        
        # if index and middle fingers both are up then selection mode

        if fingers[1] and fingers[2]:

            # changing the previous coordinates in order to start the drawing after ending of selection mode
            x_prev,y_prev = 0,0

            # print(" Both fingers up hence selection mode ")
            cv2.line(img,(x_index,y_index), (x_middle,y_middle), draw_color,brush_thickness)
            # traveling into the header
            if y_index < 125:
                # print(x_index)
                if x_index > 140 and x_index < 190:
                    header = ui_overlay_dict['blue_active.png']
                    draw_color = (255,0,0)

                elif x_index > 400 and x_index < 450:
                    header = ui_overlay_dict['red_active.png']
                    draw_color = (0,0,255)

                elif x_index > 710 and x_index < 760:
                    header = ui_overlay_dict['green_active.png']
                    draw_color = (0,255,0)

                elif x_index > 1020 and x_index < 1110:
                    header = ui_overlay_dict['eraser_active.png']
                    draw_color = (0,0,0)
                


        # if only index finger is up then drawing mode
        
        if fingers[1] and not fingers[2]:
            # print(" Only index finger up hence drawing mode ")
            cv2.circle(img,(x_index,y_index),10,draw_color,cv2.FILLED)
            if x_prev == 0 and y_prev == 0:
                x_prev,y_prev = x_index,y_index

            if draw_color == (0,0,0):
                cv2.line(paint_canvas,(x_prev,y_prev),(x_index,y_index),draw_color,eraser_thickness)
            
            else:
                cv2.line(paint_canvas,(x_prev,y_prev),(x_index,y_index),draw_color,brush_thickness)

            x_prev,y_prev = x_index,y_index


        
    # convetring the canvas image to grayscale image
    gray_img = cv2.cvtColor(paint_canvas,cv2.COLOR_BGR2GRAY)
    # masking the grayscale image 
    _,inverse_img = cv2.threshold(gray_img,50,255,cv2.THRESH_BINARY_INV)
    # converting the masked grayscale image to colored image
    inverse_img = cv2.cvtColor(inverse_img,cv2.COLOR_GRAY2BGR)
    # adding the masked image to the main image
    img = cv2.bitwise_and(img,inverse_img)
    # adding canvas image to the main image
    img = cv2.bitwise_or(img,paint_canvas)
        



    # fetching fps
    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time






    img[:125,:] = header

    cv2.putText(img,str(int(fps)),(10,10+215),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,2,(255,0,0),2)
    
    cv2.imshow("Virtual Paint",img)
    # cv2.imshow("Virtual Paint Canvas",paint_canvas)
    cv2.waitKey(1)