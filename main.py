import numpy as np
import cv2
cap = cv2.VideoCapture("con_vid.mp4")
lower = np.array([0, 0, 0], dtype = "uint8")
upper = np.array([15, 255, 255], dtype = "uint8")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
cv2.namedWindow("images",cv2.WINDOW_AUTOSIZE)
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # resize the frame, convert it to the HSV color space,
	# and determine the HSV pixel intensities that fall into
	# the speicifed upper and lower boundaries
    frame = cv2.resize(frame,(640,480))
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
 
	# apply a series of erosions and dilations to the mask
	# using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
 
	# blur the mask to help remove noise, then apply the
	# mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    contours, hierarchy = cv2.findContours(skinMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask) 
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        frame[y:y+h,x:x+w] = cv2.GaussianBlur(frame[y:y+h,x:x+w],(31,31),0)
 
	# show the skin in the image along with the mask
    if (ret==True):
        out.write(frame)
        cv2.imshow("images", np.hstack([frame,skin]))
        # if the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()