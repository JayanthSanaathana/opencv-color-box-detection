import cv2 as cv
import numpy as np
"""
Detects the largest red-colored object in the given frame using HSV color space.

Input Parameter:
    frame:Input BGR image from camera
    kernal:Morph Kernel for noise removal

Returns:
    -(x, y, w, h, area) of detected red box if found
    - mask used for detection
    If no box is found, returns (None, mask)
"""
def detect_red_box(frame,kernel):
    MIN_AREA = 2000 # min contour area to reduce noise,can be changed
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    # Red color range_Lower Hue
    lower1 = np.array([0, 80, 60], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    # Red color range_Higher Hue
    lower2 = np.array([170, 80, 60], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    mask1 = cv.inRange(hsv,lower1,upper1)
    mask2 = cv.inRange(hsv,lower2,upper2)
    #Mask for red color
    mask = cv.bitwise_or(mask1,mask2)
    mask = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel,iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    contours,_ = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    # Loop through all detected contours
    for c in contours:
        area = cv.contourArea(c)
        if area >= MIN_AREA and area > best_area:
            best_area = area
            best = c

    if best is None:
        return None, mask
    # Compute bounding rectangle for the detected contour
    x, y, w, h = cv.boundingRect(best)


    return (x, y, w, h, best_area), mask


vid  = cv.VideoCapture(1) # Change Video Capture number if multiple Cams
if not vid.isOpened():
    print("Video Source not found")
    exit()
#Create Morph kernel(size can be changed. (9,9)-More Aggressive,(5,5)- Balanced)
kernel = cv.getStructuringElement(cv.MORPH_RECT,(7,7))
while True:
    r,frame = vid.read()
    if not r:
        break

    #red box detection
    result, mask = detect_red_box(frame,kernel)
    if result is not None:
        x,y,w,h,area = result
        #draw red box from coordinates
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    cv.imshow("Video",frame)
    #exit
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

#close windows
vid.release()
cv.destroyAllWindows()