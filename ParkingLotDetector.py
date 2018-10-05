# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils
from imutils import contours
# construct the argument parse and parse the arguments
#Image , Row & Column Number
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
ap.add_argument("-l", "--image_ref", required=True, help="path to reference image")
ap.add_argument("-j","--row", help="number of rows in Parking Lot",type=int)
ap.add_argument("-k","--col", help="number of columns in Parking Lot",type=int)
args = vars(ap.parse_args())

#flag for detecting Car, 0 for empty 1 for occupied
flag = 0

# load the image
image = cv2.imread(args["image"])
image_ref = cv2.imread(args["image_ref"])
row = args["row"]
col = args["col"]

# define the list of Color boundaries[White/Grayish Color]
#The sttripe color which is Used in Parking lot to define blocks
lower = [180, 180, 180]
upper = [217, 217, 217]

# create NumPy arrays from the boundaries
lower = np.array(lower, dtype = "uint8")
upper = np.array(upper, dtype = "uint8")

# find the colors within the specified boundaries and apply
# the mask
mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask = mask)

# threshold the image, then perform a series of erosions +
# dilations to remove any small regions of noise
# mainly optimizing the image for better result
gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=1)
thresh = cv2.dilate(thresh, None, iterations=1)


# find contours in thresholded image
im2, conts, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

try: hierarchy = hierarchy[0]
except: hierarchy = []

#Setting min_x and min_y to image width and geight
height, width, _ = image.shape
min_x, min_y = width, height
max_x = max_y = 0

# computes the bounding box for the contour, and draws it on the image,
for conts, hier in zip(conts, hierarchy):
    (x,y,w,h) = cv2.boundingRect(conts)
    min_x, max_x = min(x, min_x), max(x+w, max_x)
    min_y, max_y = min(y, min_y), max(y+h, max_y)
    #this drawing is optional not necessary
    #if w > 80 and h > 80:
        #cv2.rectangle(image, (x,y), (x+w,y+h), (255, 0, 0), 2)
# drawing the biggest Area possible Covering all the Contours
if max_x - min_x > 0 and max_y - min_y > 0:
    cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)


# saving a copy of minimum x cordinate as it will be manipulated
min_xCopy = min_x

# creating the parking lot single box, it will be iterated through the whole block
boxWidth = int((max_x - min_x) / col)
boxHeight = int((max_y - min_y) / row)

# dividing the whole block into sub regions from the row column value thus getting
# each parking slot seperately
for y in range (row):
	for x in range (col):
		# setting flag to 0
		flag = 0
		# Reference Image
		cropped_ref = image_ref[min_y : (min_y+boxHeight), min_x: (min_x + boxWidth)]
		cropped_ref = cv2.cvtColor(cropped_ref, cv2.COLOR_BGR2GRAY)
		cropped_ref = cv2.GaussianBlur(cropped_ref, (5, 5), 0)

		# Test Image
		cropped = image[min_y : (min_y+boxHeight), min_x: (min_x + boxWidth)]
		crop = cropped.copy()
		cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
		cropped = cv2.GaussianBlur(cropped, (5, 5), 0)

		# Background Subtaraction
		difference = cv2.absdiff(cropped_ref, cropped)
		_, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)

		# Prepare For Contour Detection
		edged = cv2.Canny(difference, 50, 100)
		edged = cv2.dilate(edged, None, iterations=1)
		edged = cv2.erode(edged, None, iterations=1)

		# Contour Detection
		cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		if not cnts:
		    flag = 0
		else:
		    (cnts, _) = contours.sort_contours(cnts)
		    for c in cnts:
		        # if the contour is not sufficiently large, ignore it
		        if cv2.contourArea(c) > 40:
		            flag = 1



		# show the images
		cv2.imshow("Process", np.hstack([cropped_ref,cropped,edged]))
		cv2.imshow("test_image",crop)
		print(flag)
		cv2.waitKey(0)
		# update x coordinate 
		min_x = min_x + boxWidth

	#update x and y cordinate, add height with y and retrieve first min value of x	
	min_x = min_xCopy	
	min_y = min_y + boxHeight



cv2.waitKey(0)