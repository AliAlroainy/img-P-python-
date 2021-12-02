import cv2
import numpy as np
from matplotlib import pyplot as plt

img = 'm.png'


  
    
def erosionProcess(img):
    imgBinary = cv2.imread(img, 0)
    ImgGray = cv2.cvtColor(imgBinary, cv2.COLOR_GRAY2RGB)  # RGB
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(ImgGray, kernel, iterations=5)
    plt.subplot(121), plt.imshow(ImgGray), plt.title('Original')
    plt.subplot(122), plt.imshow(erosion), plt.title('Erosion')
    plt.show()



def dilationProcess(img):
    imgBinary = cv2.imread(img, 0)
    ImgGray = cv2.cvtColor(imgBinary, cv2.COLOR_GRAY2RGB)  # RGB
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(ImgGray, kernel, iterations=5)
    plt.subplot(121), plt.imshow(ImgGray), plt.title('Original')
    plt.subplot(122), plt.imshow(dilation), plt.title('Dilation')
    plt.show()

erosionProcess(img)
dilationProcess(img)
# erosionProcess('slimyGirl.png')
# dilationProcess('slimyGirl.png')


def thresholds(img):
    imgBinary = cv2.imread(img, 0)
    ImgGray = cv2.cvtColor(imgBinary, cv2.COLOR_GRAY2RGB)  # RGB
    ret, thresh1 = cv2.threshold(ImgGray, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(ImgGray, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(ImgGray, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(ImgGray, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(ImgGray, 127, 255, cv2.THRESH_TOZERO_INV)
    print('ddd', thresh1)
    titles = ['Original Image', 'BINARY',
              'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [ImgGray, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()


thresholds(img)
# thresholds('slimyGirl.png')

def hist(img):
    img1 = cv2.imread(img,0)
    hist = cv2.calcHist([img1],[0],None,[256],[0,256])
    plt.plot(hist)
    v = cv2.equalizeHist(img1)
    plt.plot(v)
    res = np.hstack((img1, v))
    cv2.imwrite('hist.png',res)
    plt.show()
    cv2.waitKey(0)

hist(img)  


  

def segmentation(imgs):
    image = cv2.imread(imgs)
    orig_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
   # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(
        dist_transform, 0.7*dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    # cv2.imshow('dd', image)
    plt.subplot(2, 3, 1), plt.imshow(thresh), plt.title('Threshold')
    plt.subplot(2, 3, 2), plt.imshow(opening), plt.title('Opening')
    plt.subplot(2, 3, 3), plt.imshow(
        dist_transform), plt.title('dist_transform')
    plt.subplot(2, 3, 4), plt.imshow(sure_fg), plt.title('sure_fg')
    plt.subplot(2, 3, 5), plt.imshow(markers), plt.title('markers')
    plt.subplot(2, 3, 6), plt.imshow(image), plt.title('Final image')
    plt.show()
    cv2.imshow('Final result', image)
    cv2.waitKey()


segmentation('m.png')

def avrageFilter(img):
    image = cv2.imread(img)
    ImgColor = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    blur = cv2.blur(ImgColor, (5, 5))
    plt.subplot(121), plt.imshow(ImgColor), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()


def gaussianFilter(img):
    image = cv2.imread(img)
    ImgColor = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    blur = cv2.GaussianBlur(ImgColor, (5, 5), 0)
    plt.subplot(121), plt.imshow(ImgColor), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    cv2.imshow('before', ImgColor)
    cv2.imshow('after', blur)
    plt.show()


def gaussianFilter(img):
    image = cv2.imread(img)
    ImgColor = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    blur = cv2.medianBlur(ImgColor, 5)
    plt.subplot(121), plt.imshow(ImgColor), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    cv2.imshow('before', ImgColor)
    cv2.imshow('after', blur)
    plt.show()

avrageFilter(img)
gaussianFilter(img)
gaussianFilter(img)

#avrageFilter('m.png')
#gaussianFilter('m.png')
#gaussianFilter('m.png')


def edgeDetection(img):
    image = cv2.imread(img, 0)
    edges = cv2.Canny(image, 90, 250)
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

edgeDetection(img)
   
def oprtion(img):
    img1 = cv2.imread(img)
    d = img1 + 100
    cv2.imshow('oprtion', d)
    cv2.waitKey(0)
    cv2.destoryAllWindows()
    
oprtion('m.png')


#print("opration")
#print("erosionProcess")
#erosionProcess('m.png')
#print("dilationProcess")
#dilationProcess('m.png')
#print("thresholds")
#thresholds('m.png')
#print("segmentation")
#segmentation('m.png')