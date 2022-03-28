import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

path0 = 'Images/test/1 ml(0.5ml kan)/' 
#20190828_111433.jpg
#20190418_094633.jpg

path1 = 'Images/test/7 ml(3.5ml kan)/'
#20190828_115509.jpg
#20190418_105801.jpg

path2 = 'Images/test/9 ml(4.5ml kan)/'
#20190828_120744.jpg
#20190418_111400.jpg

red = np.uint8([[[0,0,255]]])
hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)

##Read the image
img = cv2.imread(path2 + '20190418_111400.jpg',1)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

RED_MIN = np.array([0, 150, 150],np.uint8)
RED_MAX = np.array([20, 255, 255],np.uint8)

RED_HIGH_MIN = np.array([160,150,150],np.uint8)
RED_HIGH_MAX = np.array([180,255,255],np.uint8)

y_lower = 800
y_upper = 3900
x_lower = 200
x_upper = 1900

mask0 = cv2.inRange(img_hsv, RED_MIN, RED_MAX)
mask1 = cv2.inRange(img_hsv, RED_HIGH_MIN, RED_HIGH_MAX)

threshed = mask0 + mask1
#thresholding and blurring
blur = cv2.medianBlur(threshed,9)

#Apply Laplacian edge detection
laplacian = cv2.Laplacian(blur, cv2.CV_8UC1)


#Find the pixels where the result is bigger than N value
pts = np.argwhere(laplacian > 0)
y1,x1 = pts.min(axis = 0)
y2,x2 = pts.max(axis = 0)

y1og = y1
y2og=y2
x1og=x1
x2og=x2

#y1 > 800  y2 > 3900
#x1 > 200  x2 > 1700


error = False
iteration = 0
while iteration < 500:
    iteration+=1
    print(error)
    print("Iteration: ", iteration)
    if(x1 > x_lower):
        x1 -= 5
        print("x1 is higher than x_lower")
    if(x2 < x_upper):
        x2 +=5
        print("x2 is lower than x_higher")
    if(y1 > y_lower):
        y1 -=20
        print("y1 is higher than y_lower")
    if(y2 < y_upper):
        y2 +=2
        print("y2 is lower than y_higher")
        
    print("x1 is {x1}, x2 is {x2}, y1 is {y1}, y2 is {y2}".format(x1=x1,x2=x2,y1=y1,y2=y2))


cropped = img[y1:y2, x1:x2]
cv2.imwrite("cropped.png", cropped)

tagged = cv2.rectangle(img.copy(), (x1,y1), (x2,y2), (0,255,0), 3, cv2.LINE_AA)


resize = ResizeWithAspectRatio(tagged, height=960)

cv2.imshow("image",resize)
cv2.waitKey(0)
