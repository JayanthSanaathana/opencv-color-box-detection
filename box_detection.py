import cv2 as cv
import numpy as np



def region_of_intrest_box(video):
    ret, frame = video.read()
    if not ret:
        raise RuntimeError("Could not fetch video from camera")
    roi_box = cv.selectROI("Select Region of Intrest :",frame,showCrosshair=True)

    x, y, w, h = map(int, roi_box)
    if w == 0 or h == 0:
        raise RuntimeError("ROI not selected properly")

    return x, y, x + w, y + h

def detect_red_box(frame,coordinates,kernel):
    MIN_AREA = 2000
    x1,y1,x2,y2 = coordinates
    roi = frame[y1:y2,x1:x2]
    hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
    lower1 = np.array([0, 80, 60], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)

    lower2 = np.array([170, 80, 60], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)

    mask = cv.bitwise_or(cv.inRange(hsv,lower1,upper1),cv.inRange(hsv,lower2,upper2))
    mask = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel,iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    contours,_ = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for c in contours:
        area = cv.contourArea(c)
        if area >= MIN_AREA and area > best_area:
            best_area = area
            best = c

    if best is None:
        return None, mask

    bx, by, bw, bh = cv.boundingRect(best)


    return (bx + x1, by + y1, bw, bh, best_area), mask

vid  = cv.VideoCapture(1)
if not vid.isOpened():
    print("Video Source not found")
    exit()
roi_coordinates = region_of_intrest_box(vid)
kernel = cv.getStructuringElement(cv.MORPH_RECT,(7,7))
while True:
    r,frame = vid.read()
    if not r:
        break

    x1,y1,x2,y2 = roi_coordinates
    cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
    result, mask = detect_red_box(frame,roi_coordinates,kernel)
    if result is not None:
        bx,by,bw,bh,area = result
        cv.rectangle(frame,(bx,by),(bx+bw,by+bh),(0,255,0),3)
    cv.imshow("Video",frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        roi_coordinates = region_of_intrest_box(vid)

vid.release()
cv.destroyAllWindows()
