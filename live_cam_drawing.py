import cv2
import time
import os
import Hand_Tracking_Module as Htm
import numpy as np
import webbrowser


# drawing elements

brush_thickness = 15
draw_color = (255, 0, 0)
eraser_thickness = 100
browser_flag = True


folder_path = 'headers'
my_list = os.listdir(folder_path)

# create list of images
overlay_list = []
for img_path in my_list:
    img = cv2.imread(f'{folder_path}/{img_path}')
    overlay_list.append(img)

header = overlay_list[0]


cam_width, cam_height = 1280, 720
cap = cv2.VideoCapture(0)  # 0 is device Number
cap.set(3, cam_width)
cap.set(4, cam_height)


detector = Htm.handDetector(detectionCon=0.90)
xp, yp = 0, 0
img_canvas = np.zeros((720, 1280, 3), np.uint8)


# finger Ids
tip_Ids = [4, 8, 12, 16, 20]


while True:
    # 1- Import the image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    # 2- find the hand
    img = detector.findHands(img)
    land_mark_list = detector.findPosition(img, draw=False)

    if len(land_mark_list[0]) != 0:

        # index and middle fingers
        x1, y1 = land_mark_list[0][8][1:]
        x2, y2 = land_mark_list[0][12][1:]

    # 3- check fingers
        fingers = []
        # for thumb
        if land_mark_list[0][tip_Ids[0]][1] < land_mark_list[0][tip_Ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # for fingers
        for id in range(1, 5):
            if land_mark_list[0][tip_Ids[id]][2] < land_mark_list[0][tip_Ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # 4-Selection Mode : Two fingers are up
        if fingers[1] and fingers[2]:
            # set zero whenever hand is on selection mode
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), draw_color, cv2.FILLED)
        if y1 < 100:
            # Displaying the image
            if 0 < x1 < 100:
                # Open My website
                if browser_flag:
                    webbrowser.open("https://ht21992.pythonanywhere.com")
                    browser_flag = False
            elif 100 < x1 < 250:
                # Choose red color
                header = overlay_list[3]
                draw_color = (0, 0, 255)
            elif 260 < x1 < 350:
                # Choose blue color
                header = overlay_list[0]
                draw_color = (255, 0, 0)
            elif 360 < x1 < 450:
                # Choose green color
                header = overlay_list[1]
                draw_color = (0, 255, 0)
            elif 460 < x1 < 640:
                # Choose eraser
                header = overlay_list[2]
                draw_color = (0, 0, 0)

        # 5-Drawing Mode : Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if draw_color == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), draw_color, eraser_thickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, eraser_thickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), draw_color, brush_thickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
            xp, yp = x1, y1

    imgGray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inverse = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)

    img_inverse = cv2.cvtColor(img_inverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inverse)
    img = cv2.bitwise_or(img, img_canvas)

    # setting the toolbar image
    img[0:120, 0:647] = header
    #

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
