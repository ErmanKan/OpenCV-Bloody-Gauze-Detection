import cv2
import numpy as np
import imutils

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

path0 = 'Half_Blood\_50 kan'
path1 = 'Half_Blood\İkci seri\_50 kan (2. grup)'
path2 = 'Half_Blood\Yari kan'



##Read the image
img = cv2.imread('Half_Blood/Yari kan/12 ml(6ml kan)/20190418_114419.jpg')
img = img[300:,:]

##Convert to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


ret, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
blur = cv2.GaussianBlur(thresh,(5,5),0)

# Perform morphology
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
image_close = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel, iterations = 1)

#Apply canny edge detection
canny = cv2.Canny(image_close,30,100)
auto_canny = auto_canny(image_close)

#Apply Laplacian edge detection
laplacian = cv2.Laplacian(blur, cv2.CV_8UC1)

#Apply Sobel Edge Detection x axis
sobelx = cv2.Sobel(blur,cv2.CV_8UC1,1,0,ksize=5)
#Apply Sobel Edge Detection y axis
sobely = cv2.Sobel(blur,cv2.CV_8UC1,0,1,ksize=5)

#contours
contours, hierarchy = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
x,y,w,h = cv2.boundingRect(cntsSorted[-1])
imglol = img.copy()
crop = imglol[y: y+h, x: x+w]
cv2.fillPoly(image_close, cntsSorted[-1], [255,255,255])
cv2.drawContours(imglol, cntsSorted[-1], -1, (0,255,0), thickness = 2,lineType = cv2.LINE_AA)

"""
cnts = cv2.findContours(image_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

"""

#Find the edges where the size is bigger than zero
pts = np.argwhere(image_close == 0)
y1,x1 = pts.min(axis = 0)
y2,x2 = pts.max(axis = 0)

cropped = img[y1:y2, x1:x2]
cv2.imwrite("cropped.png", cropped)

tagged = cv2.rectangle(img.copy(), (x1,y1), (x2,y2), (0,255,0), 3, cv2.LINE_AA)


resize = ResizeWithAspectRatio(image_close, height=960)
cv2.imshow("image",resize)
cv2.waitKey(0)

"""
dsize = (124, 124)
thresh_array = []
lower = np.array([0, 0, 66])
upper = np.array([91, 243, 255])

image = cv2.imread('Half_Blood/Yari kan/1 ml(0.5ml kam)/20190418_114436.jpg')
mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask=mask)
img_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.resize(thresh, dsize)
thresh = thresh.reshape(-1)
thresh_array.append(thresh)

resize = ResizeWithAspectRatio(img_gray, height=960)
cv2.imshow("image",resize)
cv2.waitKey(0)
"""