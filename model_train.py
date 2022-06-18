import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle

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

def showImage(image, height, window_name):
    resized = ResizeWithAspectRatio(image,height = height)
    cv2.imshow(window_name,resized)
    
def showImage1(image, height, window_name):
    resized = ResizeWithAspectRatio(image,height = height)
    cv2.imshow(window_name,resized)
    
def showImage2(image, height, window_name):
    resized = ResizeWithAspectRatio(image,height = height)
    cv2.imshow(window_name,resized)
    cv2.waitKey(0)
    
def getImage(image_path):
    image = cv2.imread(image_path,1)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    return image

def getMask(mask_lower, mask_upper,image):
    mask = cv2.inRange(image, mask_lower, mask_upper)
    return mask

def applyMask(image,mask):
    masked_image = cv2.bitwise_and(image, image, mask)
    return masked_image

def hu_moments(image):
    hu_moments = cv2.HuMoments(cv2.moments(image)).flatten()
    return hu_moments

def generate_histogram(image):
    hist = cv2.calcHist([image],[0],None, [32],[0,256])
    cv2.normalize(hist,hist)
    return hist.flatten()

def getManualEdges(laplacian,THRESH,MIN_REQ):
    

    final_edges = np.zeros_like(laplacian)
    CURRENT_REQ = 0
    
    for row in range(len(laplacian[0])):
        CURRENT_REQ = 0
        
        for column in range(len(laplacian)):
            
            ##Sol
            try:
                if(abs(laplacian[column, row] - laplacian[column + 0, row - 1 ]) >= THRESH):
                    CURRENT_REQ += 1 
            except:
                pass
                    
                
            ##Sağ
            try:
                if(abs(laplacian[column, row] - laplacian[column + 0, row + 1] >= THRESH)):
                    CURRENT_REQ += 1
            except:
                pass
                
            ##Aşağı
            try:
                if(abs(laplacian[column, row] - laplacian[column + 1, row + 0] >= THRESH)):
                    CURRENT_REQ += 1
            except:
                pass
                
            ##Yukarı
            try:
               if(abs(laplacian[column, row] - laplacian[column - 1, row + 0 ] >= THRESH)):
                   CURRENT_REQ += 1
            except:
                pass
                
            ##Sol Aşağı
            try:
                if(abs(laplacian[column, row] - laplacian[column + 1, row - 1 ] >= THRESH)):
                    CURRENT_REQ += 1
            except:
                pass
                
            ##Sağ aşağı
            try:
                if(abs(laplacian[column, row] - laplacian[column + 1, row + 1 ] >= THRESH)):
                    CURRENT_REQ += 1
            except:
                pass
                
            ##Sol Yukarı
            try:
                if(abs(laplacian[column, row] - laplacian[column - 1, row - 1 ] >= THRESH)):
                    CURRENT_REQ += 1
            except:
                pass
                
            ##Sağ Yukarı
            try:
                if(abs(laplacian[column, row] - laplacian[column - 1, row + 1 ] >= THRESH)):
                    CURRENT_REQ += 1
            except:
                pass
                 
            try:
                if(CURRENT_REQ >= MIN_REQ):
                    final_edges[column, row] = 1
            except:
                pass
    return final_edges

def uniqueLaplacianValues(laplacian):
    
    unique = np.unique(laplacian)
    values, counts = np.unique(laplacian, return_counts=True)
    combined = np.stack([values, counts], axis = 1)
    return combined

def tagArea(laplacian,THRESH):
    try:
        pts = np.argwhere(laplacian >= THRESH)
        y1,x1 = pts.min(axis = 0)
        y2,x2 = pts.max(axis = 0)
    
        tagged = cv2.rectangle(img.copy(), (x1,y1), (x2,y2), (0,255,0), 3, cv2.LINE_AA)
        tagged = cv2.cvtColor(tagged,cv2.COLOR_HSV2BGR)
        cropped = img[y1:y2, x1:x2]
        cropped = cv2.cvtColor(cropped,cv2.COLOR_HSV2BGR)
        cv2.imwrite("cropped.png", cropped)
        return tagged
    except ValueError:
        return "Error"

    
path0 = 'Images/test/1 ml(0.5ml kan)/' 
#20190828_111433.jpg
#20190418_094633.jpg

path1 = 'Images/test/7 ml(3.5ml kan)/'
#20190828_115509.jpg
#20190418_105801.jpg

path2 = 'Images/test/9 ml(4.5ml kan)/'
#20190828_120744.jpg
#20190418_111400.jpg

path3 = 'Images/test/8 ml(4ml kan)/'
#20190418_110648.jpg

path4 = 'Images/test/12 ml(6ml kan)/'
#20190828_122924.jpg

BLUE_UPPER = np.array([110,255,255])
BLUE_LOWER = np.array([83,0,0])
THRESH = 7
MIN_REQ = 3

"""
img = getImage(path4 + '20190828_122924.jpg')
mask = getMask(BLUE_LOWER, BLUE_UPPER, img)
masked = applyMask(img,mask)
blur = cv2.GaussianBlur(masked,(9,9),0)
imgBGR = cv2.cvtColor(blur,cv2.COLOR_HSV2BGR_FULL)
gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
laplacian = cv2.Laplacian(gray,cv2.CV_64F, ksize = 1)
feature = hu_moments(laplacian)
hist = generate_histogram(gray)
"""

###############################################################
"""
#final_edges = getManualEdges(laplacian, THRESH, MIN_REQ)
tagged = tagArea(laplacian,THRESH)
showImage(laplacian,960,"laplacian")
#showImage(final_edges,960,"manual edges")
showImage(masked,960,"masked")
if(tagged != "error"):
    showImage2(tagged,960,"cropped original")
"""
###############################################################

label_count = 12
images_per_class = 15
label = 0
train_path = "Images\\train"
test_path = "Images\\test"
train_labels = ["1 ml(0.5ml kan)", "2 ml(1ml kan)", "3 ml(1.5ml kan)", "4 ml(2ml kan)", "5 ml(2.5ml kan)", "6 ml(3ml kan)" , "7 ml(3.5ml kan)", "8 ml(4ml kan)" ,"9 ml(4.5ml kan)", "10 ml(5ml kan)", "11 ml(5.5ml kan)", "12 ml(6ml kan)"]

global_features = []
labels = []

h5_data = 'output/data.h5'
h5_labels = 'output/labels.h5'

for label in train_labels:
    dir = os.path.join(train_path, label)
    print(dir)
    current_label = label
    
    for count in range(1, images_per_class+1):
        folder = dir + "\\" + "image_"
        file = folder  + "("+ str(count) + ")" + ".jpg"
        print(file)
        img = getImage(file)
        mask = getMask(BLUE_LOWER, BLUE_UPPER, img)
        masked = applyMask(img,mask)
        blur = cv2.GaussianBlur(masked,(9,9),0)
        imgBGR = cv2.cvtColor(blur,cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray,cv2.CV_64F, ksize = 1)
        
        feature = hu_moments(gray)
        hist = generate_histogram(gray)
        
        global_feature = np.hstack([hist,feature])
        labels.append(current_label)
        global_features.append(global_feature)
        
    print("processed folder: {}".format(current_label))
    
targetNames = np.unique(labels)
encoder = LabelEncoder()
target = encoder.fit_transform(labels)

scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("feature vector normalized...")

print("target labels: {}".format(target))
print("target labels shape: {}".format(target.shape))

# save the feature vector using HDF5
seed = 0
results = []
names   = []
models = []

models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier(random_state=seed)))
models.append(('RF', RandomForestClassifier(n_estimators=100, random_state=seed)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=seed)))

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(rescaled_features),np.array(labels),test_size=0.33,random_state=seed)

# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(model, trainData, trainLabels, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


filename = 'rfc_model.sav'
clf = SVC(random_state=seed)
clf.fit(trainData,trainLabels)
pickle.dump(clf, open(filename, 'wb'))



