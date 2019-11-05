import cv2
import numpy as np
import Deskew
from matplotlib import pyplot as plt


def detectPlatesRectangle(thresholdedImage, originalImg, verbose):
#Contours detection    
    _, contours, hierarchy = cv2.findContours(thresholdedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    originalCopy = originalImg.copy()
    possiblePlateNum = 0
    possiblePlates = []
    for contour in contours:
        # get rectangle bounding contour
        [x,y,w,h] = cv2.boundingRect(contour)
        # discard areas that are too large
        if h>100 and h<40:
            continue
        # discard areas that are too small
        if w>250 or w<120:
            continue

        if h<100 and h>40 and w<250 and w>120:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(originalCopy, [box], 0, (0,0,255), 4)
            # draw rectangle around contour on original image
            possiblePlateNum = possiblePlateNum + 1
            roi = originalImg[y:y+h, x:x+w]
            roi_t = cv2.bitwise_not(thresholdedImage)[y:y+h, x:x+w]
            possiblePlates.append(roi_t)
            cv2.imwrite( "./PossiblePlates/rectangle_bitwisenot" + str(possiblePlateNum) + ".png", roi_t)
            cv2.imwrite( "./PossiblePlates/rectangle" + str(possiblePlateNum) + ".png", roi)
        #Draw rectangle without rotation    
            #cv2.rectangle(thresholdedImage, (x,y), (x+w,y+h),(255,0,255),4)
    connectedComponents(possiblePlates, verbose, 'rect')
    print("Number of possible plates found: " + str(len(possiblePlates)))
# write original image with added contours to disk
    if verbose:
        cv2.imshow("Candidates", originalCopy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def detectPlatesHAAR(originalImg, verbose):
    plateCascade = cv2.CascadeClassifier('./cascade.xml')
    gray = cv2.cvtColor(originalImg, cv2.COLOR_BGR2GRAY)
    plates = plateCascade.detectMultiScale(gray, 1.8, 5)
    originalCopy = originalImg.copy()
    possiblePlates = []

    print(len(plates))
    print(plates)
    if len(plates) != 0:
       for (x,y,w,h) in plates:
           cv2.rectangle(originalCopy,(x,y),(x+w,y+h),(255,0,0),2)
           roi_gray = gray[y:y+h+5, x:x+w+5]
           roi_color = originalImg[y:y+h, x:x+w]
       imgBlurred = cv2.GaussianBlur(roi_gray, (3,3) ,0)
       ret, otsu = cv2.threshold(imgBlurred, 0, 255.0, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
       cv2.imwrite("./PossiblePlates/haar.png", otsu)
       possiblePlates.append(otsu)
    
       if verbose:
           cv2.imshow("HAAR detection", originalCopy)
           cv2.imshow("Otsu plate", otsu)
           cv2.waitKey(0)
           cv2.destroyAllWindows()
       connectedComponents(possiblePlates, verbose, 'haar')

def connectedComponents(imgArray, verbose, method):
    path = method+'/' 
    plateNum = 0
    for img in imgArray:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)
        stats = sorted(stats, key=lambda x: x[0]) 
        stats = np.asarray(stats)
        imgCopy = img.copy()
        heightArr = stats[:, 3]
        heightArr = heightArr[heightArr>5]
        counts = np.bincount(heightArr)
        approx = np.argmax(counts)
        possibleChars = []
        
        for i in range(num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            a = stats[i, cv2.CC_STAT_AREA]
            if a < 30 or a > 400:
                continue
            if h > approx + 3: 
                continue 
            if h < approx:
                continue
            if w > 55:
                continue
            if x == 0 or y == 0:
                continue
            roi = img[y:y+h, x: x+w]
            possibleChars.append(roi)
            cv2.rectangle(imgCopy, (x,y), (x+w, y+h), (255,0,255), 1)
        if verbose:
            cv2.imshow("Possible characters", imgCopy)    
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        for j in range(len(possibleChars)):
            cv2.imwrite("./PossibleChars/" + path + str(j) + ".png", possibleChars[j])
