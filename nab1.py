import cv2
import numpy as np

cap = cv2.VideoCapture(0)
draw = np.zeros((700,800,3), np.uint8)
while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    blurred = cv2.GaussianBlur(blurred, (11, 11), 0)
    blurred = cv2.GaussianBlur(blurred, (11, 11), 0)
    blurred = cv2.GaussianBlur(blurred, (11, 11), 0)
    blurred = cv2.GaussianBlur(blurred, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (11, 11), 0)

    # define range of blue color in HSV
    lower_blue = np.array([160, 100, 100])
    upper_blue = np.array([179, 255, 255])
    # Threshold the HSV image to get only blue colors

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blurred1 = cv2.GaussianBlur(mask, (11, 11), 0)
    blurred1 = cv2.GaussianBlur(blurred1, (11, 11), 0)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    #ret,thresh = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    ret,thresh = cv2.threshold(blurred1,0,255,cv2.THRESH_BINARY)

    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #img = cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    x1=int(700-(x+(w/2)))
    y1=int(y+(h/2))
    #cv2.line(frame,(0,0),(511,511),(255,0,0),5)
    
    draw = cv2.line(draw,(x1,y1),(x1,y1),(255,0,0),5)
    #draw = cv2.line(img,(x1,y1),(x1,y1),(255,0,0),5)
    cv2.imshow('Draw',draw)
    cv2.imshow('frame',img)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()