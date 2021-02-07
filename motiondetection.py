import cv2
import numpy as np

capture = cv2.VideoCapture('test2.mp4')

reference = None

#while video in capture is playing.
while capture.isOpened():

    #store current frame being viewed
    frame = capture.read()[1]


    # convert current frame to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur the image to for differentiating purposes
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    #if reference frame is none, initalize
    if reference is None:
        reference = gray.copy().astype("float")

    #accumulate the reference image as a weighted average, so
    #for small variables in image to not be tracked as much
    cv2.accumulateWeighted(gray, reference, 0.5)

    #difference between average and current frame
    change = cv2.absdiff(gray, cv2.convertScaleAbs(reference))


    # threshold for a difference to be recognized in motion
    thresh = cv2.threshold(change, 20, 255, cv2.THRESH_BINARY)[1]

    #iterations is noise level, change to higher for less noise
    dilate = cv2.dilate(thresh, None, iterations=3)

    #contours
    contours = cv2.findContours(dilate.copy(), cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)[0]

    # highlight motion
    for c in contours:
        #noise level once again in terms of pixels
        if cv2.contourArea(c) >= 600:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display frame
    cv2.imshow('Motion Detector', frame)


    # Press 'esc' for quit
    if cv2.waitKey(40) == 27:
        break

#close calls
capture.release()
cv2.destroyAllWindows()
