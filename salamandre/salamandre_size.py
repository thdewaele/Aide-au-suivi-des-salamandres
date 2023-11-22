#Generate by chatGP on 30/01/2023

import cv2
import numpy as np

# Load the image
img = cv2.imread("IMG_0983.jpg")

# Convert the image to RGB
RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Convert the image to grayscale
gray = cv2.cvtColor(RGB, cv2.COLOR_RGB2GRAY)

#Difference of Gaussian
low_sigma = cv2.GaussianBlur(img,(3,3),0)
high_sigma = cv2.GaussianBlur(img,(5,5),0)
dog = low_sigma -high_sigma

#Filtre jaune
jaune = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
loweryellow = np.array([22,93,0], dtype="uint8")
upperyellow = np.array([45,255,255], dtype="uint8")

mask = cv2.inRange(jaune,loweryellow,upperyellow)

cnts = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) ==2 else cnts[1]



#Set our filtering parameters/ Intialize parameter setting using cv2.SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

#Set Area filtering parameters
params.filterByArea = True
params.minArea = 400

#Set circularity filtering parameters
params.filterByCircularity = True
params.minCircularity = 0.75

#Set convexity
params.filterByConvexity = True
params.minConvexity = 0.2

#Set inertia filtering parameters
params.filterByInertia = True
params.minInertiaRatio = 0.01

#Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

#Detect blods
coin = detector.detect(gray)

# Draw blobs on our image as red circles
#blank = np.zeros((10, 10))
cv2.drawKeypoints(img, coin, img, (0, 0, 255),10)

number_of_blobs = len(coin)
text = "Number of Circular Blobs: " + str(len(coin))
cv2.putText(img, text, (20, 550),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

# Apply thresholding to segment the coin
#_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
edges = cv2.Canny(jaune, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Find the contour of the 1 euro
for contour in contours:
    if cv2.contourArea(contour) > 400:
        coin_contour = contour
        break

# Calculate the size of the coin in pixels
coin_size = cv2.contourArea(coin_contour)

# Draw the contour of the 1euro  on the image in red
#img = cv2.drawContours(img, [coin_contour], 0, (0, 255, 0), 10)

# Find the contour of the object you want to measure
for contour in contours:
    if cv2.contourArea(contour)> 600 and not np.all(contour == coin_contour):
        object_contour = contour
        break


# Calculate the size of the object in pixels
object_size = cv2.contourArea(object_contour)
"""
# Find the contour of the coin
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
coin_contour = max(contours, key=cv2.contourArea)

# Fit an ellipse around the contour
ellipse = cv2.fitEllipse(coin_contour)

# Use the known diameter of the coin to calculate the scale factor
coin_diameter = 23 # in millimeters
pixel_per_mm = ellipse[1][0] / coin_diameter

# Find the contour of the object
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
object_contour = max(contours, key=cv2.contourArea)

# Measure the size of the object
object_size = cv2.arcLength(object_contour, True) / pixel_per_mm

"""
# Draw the contours on the image
img = cv2.drawContours(img, [coin_contour], 0, (0, 0, 255), 10)
img = cv2.drawContours(img, [object_contour], 0, (255, 0, 0), 10)
img = cv2.drawKeypoints(img, coin, img, (0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



# Show the image


# Create a named window and specify its size
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 500, 900)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Longueur de la salamande : ", object_size/10, "cm")

"""
#base on the explanation of https://pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def midpoint(p1, p2):
    return ((p1[0] +p2[0]) * 0.5, (p1[1] + p2[1]) *0.5)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help = "path to the input image")
ap.add_argument("-w", "--width", type=float, required=True, help = "width of the left most object in the image (in cm)")
args = vars(ap.parse_args())


#load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(7,7),0)

#perfom edge detection, then perform a dilation + erosion to close gaps in between object edges

edged = cv2.Canny(gray, 50,100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

#find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#sort the contours from left-to-rigth and intialise the  'pexils per metric' colibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None


#loop over the contours individually
for c in cnts:
    #if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 100:
        continue

    #compute the rotated bounding box of the contour
    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    #order the points in the contour such that they appear in top left, top-rigth, bottom-rigth, bottom-left order, then drax the outline of the rotated bounding box
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1,(0,255,0), 2)

    #loop over the orginal points and draw them
    for (x,y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0,0,255),-1)


    #unpack the ordered bounding box, then compute the midpoint between the top-left and top-right coordinates, followed by the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    #draw the midpoints on the image
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    #draw lines between the midpoints
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)

    #compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # if the pixels per metric has not been initialized, then compute it as the ratio of pixels to supplied metric (in this case, cm)
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / args["width"]

    # compute the size of the object
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric
    # draw the object sizes on the image
    cv2.putText(orig, "{:.1f}in".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}in".format(dimB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    # show the output image
    cv2.imshow("Image", orig)
    cv2.waitKey(0)
    """