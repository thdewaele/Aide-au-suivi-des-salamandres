
import cv2
import numpy as np
from flask import Blueprint

def get_coordPiece(filename, piece):
    img = cv2.imread(filename)
    img2 = img.copy()

    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
    template = cv2.imread('1euro.png',0)
    if (piece == 2):
        template = cv2.imread('test.png',0)
    elif(piece == 3):
        template = cv2.imread('0.5euro_det.png',0)
    rgb = cv2.cvtColor(template,cv2.COLOR_BGR2RGB)

    w,h = template.shape[::-1]

    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


    method = cv2.TM_CCORR
    if (piece==2):
        method =cv2.TM_SQDIFF
    # Apply template Matching
    res = cv2.matchTemplate(img2, rgb, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] +h)

    centre = ( int(top_left[0]+w/2), int (top_left[1]+h/2))
    print(centre)

    cv2.rectangle(img, top_left, bottom_right, (255,255,0), 5)
    cv2.circle(img, centre, 180, (255,255,0), 5)

    cv2.namedWindow("Detect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detect", 500, 900)
    cv2.imshow ('Detect', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    get_coordPiece("20230319_195622008_iOS.jpg",2)
    get_coordPiece("IMG_0962.jpg", 1)



