# import the necessary packages
import argparse
import cv2
import imutils
from imutils import contours

#Reading The Image File
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_ref", required=True,help="path to refrence image")
ap.add_argument("-j", "--image", required=True,help="path to input image")
args = vars(ap.parse_args())
image_ref = cv2.imread(args["image_ref"])
image = cv2.imread(args["image"])

#Cordinates Of Slot 1
x = 160
y = 240
x1 = 160
y1 = 240
w = 105
h = 210

#Flag for Detecting Cars
flag = 0

#Drawing Rectangle For Better Undersatnding
#cv2.rectangle(image, (x, y), (x + w, y+h), (0, 255, 0), 2)

for i in range(7):
		
	#Reference Image
	cropped_ref = image_ref[y : y+h, x : x+w]
	cropped_ref = cv2.cvtColor(cropped_ref, cv2.COLOR_BGR2GRAY)
	cropped_ref = cv2.GaussianBlur(cropped_ref, (5, 5), 0)


	#Test Image
	cropped = image[y1 : y1+h, x1 : x1+w]
	cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
	cropped = cv2.GaussianBlur(cropped, (5, 5), 0)


	#Background Subtaraction
	difference = cv2.absdiff(cropped_ref, cropped)
	_, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)


	#Prepare For Contour Detection
	edged = cv2.Canny(difference, 50, 100)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)

	#Contour Detection
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	if not cnts:
		flag = 0
	else:
		(cnts, _) = contours.sort_contours(cnts)
		for c in cnts:
			# if the contour is not sufficiently large, ignore it
			if cv2.contourArea(c) > 100:
				flag = 1
	
	x = x + w
	x1 = x1 + w
	print (flag)
	cv2.imshow("Input", cropped_ref)

	cv2.imshow("Grayed", cropped)

	cv2.imshow("Contour", edged)

	cv2.imshow("Output", crop)

	cv2.waitKey(0)

cv2.waitKey(0)