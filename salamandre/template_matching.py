import cv2
from flask import Blueprint

bp = Blueprint('template', __name__, url_prefix='')
@bp.route('/template', methods=['POST'])
def get_coordPiece(filename, piece):
    img = cv2.imread(filename)
    img2 = img.copy()

    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
    if (piece == 1):
        template = cv2.imread('1eurodet.png',0)
    elif (piece == 2):
        template = cv2.imread('2euro_det.png',0)
    elif(piece == 3):
        template = cv2.imread('0.5euro_det.png',0)
    rgb = cv2.cvtColor(template,cv2.COLOR_BGR2RGB)

    w,h = template.shape[::-1]

    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']



    method =cv2.TM_CCORR
    # Apply template Matching
    res = cv2.matchTemplate(img2, rgb, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] +h)
    cv2.rectangle(img, top_left, bottom_right, (255,255,0), 5)

    cv2.namedWindow("Detect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detect", 500, 900)
    cv2.imshow ('Detect', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    get_coordPiece("20230319_195622008_iOS.jpg",2)


"""
img = cv2.imread("IMG_0962.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread("1eurodet.jpg")
w,h = template.shape[:2]
tem_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



res = cv2.matchTemplate(image = gray, templ = tem_gray, method = cv2.TM_CCOEFF_NORMED)

threshold = 0.9


loc = np.where(res >= threshold)


for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (255, 255, 255), 5)
(y_points, x_points) = np.where(res >= threshold)

boxes= list()

for (x,y) in zip(x_points, y_points):
    boxes.append ((x,y, x+w, y +h))

#boxes = non_max_suppression(np.array(boxes))

for (x1,y1,x2,y2) in boxes:
    cv2.rectangle(img,(x1,y1),(x2,y2), (0,255,255),5)


cv2.namedWindow("Detect", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detect", 500, 900)
cv2.imshow ('Detect', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""