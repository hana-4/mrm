import cv2
import numpy as np

cap = cv2.VideoCapture(0)
#set size of video feed window
cap.set(3,1280)
cap.set(4,720)

def nothing (x):
	pass

#create trackbar window
cv2.namedWindow('Trackbars')
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:

	ret, frame = cap.read()
	if not ret:
		break
	frame = cv2.flip(frame, 1)
	hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

	l_h = cv2.getTrackbarPos("L - H", "Trackbars")
	l_s = cv2.getTrackbarPos("L - S", "Trackbars")
	l_v = cv2.getTrackbarPos("L - V", "Trackbars")
	u_h = cv2.getTrackbarPos("U - H", "Trackbars")
	u_s = cv2.getTrackbarPos("U - S", "Trackbars")
	u_v = cv2.getTrackbarPos("U - V", "Trackbars")
	#lower_range = np.array([l_h, l_s, l_v]) 
	lower_range=np.array([ 105, 150, 151])
	upper_range = np.array([u_h, u_s, u_v])
	
	mask = cv2.inRange(hsv, lower_range, upper_range)
	res = cv2.bitwise_and(frame, frame, mask = mask)

	mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

	stacked = np.hstack((mask_3, frame, res))
	cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))
	key = cv2.waitKey(1)
	if key == 10:
		break

	if key == ord('s'):
		thearray = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
		print(thearray)
		np.save('hsv_value',thearray)
		break
      
cap.release()
cv2.destroyAllWindow()

#array([  5, 150, 151])