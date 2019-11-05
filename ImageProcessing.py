import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

THRESHOLD_VALUE = 128
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)

class ImageProcessing:
    
    def __init__(self, image, verbose):
        self.image = image
        self.verbose = verbose
        self.height, self.width, self.channels = self.image.shape

    def preProcessHSV(self):
    #Prepare HSV arrays.
        imgHSV = np.zeros((self.height, self.width, 3), np.uint8)
        imgHSV = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)
        imgHue, imgSaturation, self.imgValue = cv.split(imgHSV)
        
        if self.verbose:
            cv.imshow("imgVal", self.imgValue)
    
    #TopHat & BlackHat morphological operations - exaggerating lighter portions.
        imgTopHat = np.zeros((self.height, self.width, 1), np.uint8)
        imgBlackHat = np.zeros((self.height, self.width, 1), np.uint8)

        structuringElement = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

        imgTopHat = cv.morphologyEx(self.imgValue, cv.MORPH_TOPHAT, structuringElement)
        imgBlackHat = cv.morphologyEx(self.imgValue, cv.MORPH_BLACKHAT, structuringElement)
        imgGrayscalePlusTopHat = cv.add(self.imgValue, imgTopHat)
        imgGrayscaleHSV = cv.subtract(imgGrayscalePlusTopHat, imgBlackHat)
        
        if self.verbose:
            cv.imshow("GrayscaleHSV", imgGrayscaleHSV)

        imgBlurred = np.zeros((self.height, self.width, 1), np.uint8)
    #Gaussian blurring
        self.imgBlurred = cv.GaussianBlur(imgGrayscaleHSV, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
        
        if self.verbose:
            cv.imshow("Blurred", self.imgBlurred)
        
        self.thresholdedImageHSV = cv.adaptiveThreshold(self.imgBlurred, 255.0, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
        ret, self.otsu = cv.threshold(self.imgBlurred, 0, 255.0, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        if self.verbose:
            cv.imshow("Otsu", self.otsu)
            cv.imshow("Threshold", self.thresholdedImageHSV)
            cv.waitKey(0)
            cv.destroyAllWindows()
        
        return self.thresholdedImageHSV, self.otsu, self.image

 
