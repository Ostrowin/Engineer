import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks_cwt
from math import ceil
import operator
import os

class imageProcessing:
    
    def __init__(self, path, withSaving, imgSize, val_gaussianBlur, smoothProjectionValue, widthPeaks):

        self.stepIt = 0
        self.withSaving = withSaving
        self.smoothValue = smoothProjectionValue;
        self.val_gaussianBlur = val_gaussianBlur
        self.widthPeaks = widthPeaks
        self.image = cv.imread(path, cv.IMREAD_GRAYSCALE)
        self.image = self.image[0:self.image.shape[0],0:self.image.shape[1]-30]
        self.image = cv.resize(self.image, imgSize)
        #blur
        self.blurred = cv.GaussianBlur(self.image,(val_gaussianBlur, val_gaussianBlur),0)
        #binary
        self.blackAndWhite = cv.threshold(self.blurred, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        #edges
        self.sobel = self.sobel(self.blackAndWhite)
        if self.withSaving:
            cv.imwrite('Steps/step'+str(self.stepIt)+'.png', self.blurred)
            self.stepIt = self.stepIt + 1
            cv.imwrite('Steps/step'+str(self.stepIt)+'.png', self.blackAndWhite)
            self.stepIt = self.stepIt + 1
            cv.imwrite('Steps/step'+str(self.stepIt)+'.png', self.sobel)

        #projections of an image
        self.verticalProjectionData = self.getVerticalProjectionReduce(self.sobelVertical(self.blackAndWhite), smoothProjectionValue)
        self.horizontalProjectionData = self.getHorizontalProjectionReduce(self.sobelHorizontal(self.blackAndWhite), smoothProjectionValue)
        self.c1 = 0.55
        self.c2 = 0.42
        self.verticalPeaks = self.getPeaks(self.verticalProjectionData, self.widthPeaks)
        #find peaks
        self.Peaks = self.calcPeaksFoots(self.verticalPeaks, self.verticalProjectionData, self.c1)
        #cut bands
        self.Bands = self.getBands(self.Peaks, self.image, self.verticalProjectionData)
        self.Bands2 = []
        it = 0
        for band in self.Bands:
            if band.shape[0] < 15 or band.shape[0] > 200:
                continue
            bandVertProjTMP = self.getVerticalProjectionReduce(self.sobelVertical(cv.threshold(cv.GaussianBlur(band,(val_gaussianBlur, val_gaussianBlur),0), 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]),smoothProjectionValue)
            bandVerticalPeaks = self.getPeaks(bandVertProjTMP, self.widthPeaks)
            PeaksWithFoots = self.calcPeaksFoots(bandVerticalPeaks, bandVertProjTMP, self.c2)
            tmp = self.getBands(PeaksWithFoots, band, bandVertProjTMP)
            for i in tmp:
                cv.imwrite('Bands/'+str(it)+".png", i)
                it = it + 1
                self.Bands2.append(i)
                
        self.Bands = []
        tmp = 0
        itPlate = 0        
        self.PlateCandidates = self.getPlates(self.Bands2)
        for i in self.PlateCandidates:
            cv.imwrite('Plates/'+str(itPlate)+".png", i)
            itPlate = itPlate + 1
            
        self.BestCandidates = self.heuristics(self.PlateCandidates)
        it = 0
        it2 = 0
        for bc in self.BestCandidates:
            self.Chars = self.Segmentation(bc)
            cv.imwrite("PossiblePlates/"+str(it2)+".png", bc)
            for c in self.Chars:
                exists = os.path.isdir("PossibleChars/hist/"+str(it2))
                if not exists:
                    os.mkdir("PossibleChars/hist/"+str(it2))
                if (c.shape[0] * c.shape[1]) * 0.9 < cv.countNonZero(c):
                    continue
                cv.imwrite("PossibleChars/hist/"+str(it2)+"/"+str(it)+".png", cv.bitwise_not(c))
                it = it + 1
            it2 = it2 + 1
            it = 0
            
            
    def sobel(self, s):
        sobelx64f = cv.Sobel(s, cv.CV_64F, 1, 0, ksize=3)
        sobely64f = cv.Sobel(s, cv.CV_64F, 0, 1, ksize=3)

        abs_dx = cv.convertScaleAbs(sobelx64f)
        abs_dy = cv.convertScaleAbs(sobely64f)
        abs_dx = np.uint8(abs_dx)
        abs_dy = np.uint8(abs_dy)
        return cv.addWeighted(abs_dx, 5, abs_dy, 5, 1)

    def sobelVertical(self, s):
        sobelx64f = cv.Sobel(s, cv.CV_64F, 1, 0, 5)
        abs_dx = cv.convertScaleAbs(sobelx64f)
        abs_dx = np.uint8(abs_dx)
        return abs_dx

        
    def sobelHorizontal(self, s):
        sobely64f = cv.Sobel(s, cv.CV_64F, 0, 1, 5)
        abs_dy = cv.convertScaleAbs(sobely64f)
        abs_dy = np.uint8(abs_dy)
        return abs_dy

    def getVerticalProjectionReduce(self, s, smoothValue):
        verticalProjection = 0;
        verticalProjection = cv.reduce(s, 1, cv.REDUCE_SUM, dtype=cv.CV_32S)
        return self.smoothVerticalValues(verticalProjection,smoothValue)
        
    def getHorizontalProjectionReduce(self, s, smoothValue):
        horizontalProjection = 0;
        horizontalProjection = cv.reduce(s, 0, cv.REDUCE_SUM, dtype=cv.CV_32S)
        return self.smoothHorizontalValues(horizontalProjection,smoothValue)
    
    def smoothVerticalValues(self, data, smoothValue):
        data2 =[]
        for i in data:
            data2.append(i[0])
        if smoothValue == 1:
            return data2
        smoothVal = np.ones((smoothValue,), dtype=np.int)
        smoothVal[int(len(smoothVal)/2)] = 0
        return np.convolve(data2, smoothVal, mode='same')

    def smoothHorizontalValues(self, data, smoothValue):
        if smoothValue == 1:
            return data[0]
        smoothVal = np.ones((smoothValue,), dtype=np.int)
        smoothVal[int(len(smoothVal)/2)] = 0
        return np.convolve(data[0], smoothVal, mode='same')
        
    def getPeaks(self, projection, widthPeaks):
        peaksArray = []
        for val in find_peaks_cwt(projection, np.arange(1, widthPeaks), noise_perc=10):
            peaksArray.append([val, projection[val]])
        return peaksArray
            
    def calcPeaksFoots(self, peaks, projection, c):
        peaks = sorted(peaks, key=operator.itemgetter(1, 0))
        allPeaks = []
        while peaks:
            ymax = peaks.pop()
            
            yb0 = c * ymax[1]
            yb1 = c * ymax[1]
            it = ymax[0]-1
            while True:
                if it == 0:
                    yb0 = [it, projection[it]]
                    break
                if projection[it] < yb0:
                    yb0 = [it, projection[it]]
                    break
                it = it-1
            it = ymax[0]+1
            while True:
                if it == (len(projection)-1):
                    yb1 = [it, projection[it]]
                    break
                if projection[it] < yb1:
                    yb1 = [it, projection[it]]
                    break
                it = it+1
            allPeaks.append([ymax,yb0,yb1])
            
        return allPeaks;

    def getBands(self, allPeaks, baseImage, projection):
        it = 0
        w = baseImage.shape[1] - 1
        bands = []
        projTMP = projection
        for i in allPeaks:

            if projTMP[i[0][0]] == 0:
                continue
            l = i[1][0]
            r = i[2][0]
            while l < r:
                projTMP[l] = 0
                l = l + 1
            crop_img = baseImage[i[1][0]:i[2][0],0:w]
            bands.append(crop_img)
        return self.Remove(bands)


    
    def getPlates(self, bands):
        plateCandidates = []
        it = 0
        it2 = 0
        for i in bands:
            bandBlurred = cv.GaussianBlur(i,(3,3),0)
            bandBlackAndWhite = cv.threshold(bandBlurred, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
            horizontalBandProjectionData = self.getHorizontalProjectionReduce(self.sobelHorizontal(bandBlackAndWhite),51)
            bandsPeaks = self.getPeaks(horizontalBandProjectionData, self.widthPeaks)
            bandsPeaks = sorted(bandsPeaks, key=operator.itemgetter(1, 0))
            bandsPeaks = self.calcPeaksFoots(bandsPeaks, horizontalBandProjectionData,0.86)
            h = i.shape[0]-1
            
            for cand in bandsPeaks:
                if horizontalBandProjectionData[cand[0][0]] == 0:
                    continue
                l = cand[1][0]
                r = cand[2][0]
                while l < r:
                    horizontalBandProjectionData[l] = 0
                    l = l + 1
                crop_img = i[0:h ,cand[1][0]:cand[2][0]]
                if crop_img.shape[0] < 24 or crop_img.shape[0] > 80 or crop_img.shape[1] > 250 or crop_img.shape[1] < 120 or crop_img.shape[1] < crop_img.shape[0]:
                    continue
                plateCandidates.append(crop_img)

        return self.Remove(plateCandidates)
            

    def heuristics(self, plates):
        bestCandidates = []

        for plate in plates:
            plateSkewed = self.Skew(plate)
            
            blurredPlate = cv.GaussianBlur(plateSkewed,(self.val_gaussianBlur, self.val_gaussianBlur),0)
            BAndWPlate = cv.threshold(blurredPlate, 127, 1, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

            proj = self.getVerticalProjectionReduce(self.sobelVertical(BAndWPlate), 1)

            x1 = abs((plateSkewed.shape[1]/plateSkewed.shape[0])-5)
            
            if x1 < 0.1:
                x2 = 1
            elif x1 < 0.2:
                x2 = 0.9
            elif x1 < 0.3:
                x2 = 0.8
            elif x1 < 0.4:
                x2 = 0.7
            elif x1 < 0.5:
                x2 = 0.6
            elif x1 < 0.6:
                x2 = 0.5
            elif x1 < 0.7:
                x2 = 0.4
            elif x1 < 0.8:
                x2 = 0.3
            elif x1 < 0.9:
                x2 = 0.2
            else:
                x2 = 0
            a = (0.3 * max(proj))/100 + (0.05 * sum(proj)/plateSkewed.shape[1])/10 + 0.3 * x2
            
            if len(bestCandidates) < 10:
                bestCandidates.append([round(a, 8), plateSkewed])
                bestCandidates = sorted(bestCandidates, key=operator.itemgetter(0))
            else:
                if a > bestCandidates[0][0]:
                    bestCandidates[0] = [round(a, 8), plateSkewed]
                    bestCandidates = sorted(bestCandidates, key=operator.itemgetter(0))

        bestCandidates = sorted(bestCandidates, key=operator.itemgetter(0),reverse=True)
        bestC = []
        for bc in bestCandidates:
            bestC.append(bc[1])
                
        return bestC
    
    def Remove(self, duplicate): 
        final_list = [] 
        for num in duplicate: 
            if np.all(final_list != num): 
                final_list.append(num) 
        return final_list 

    def Skew(self, plate):
        edges = cv.Canny(plate,50,150,apertureSize = 3)
        angle = 0
        lines = cv.HoughLinesP(edges,rho = 1,theta=np.pi/180,threshold=20,minLineLength=plate.shape[1]/2, maxLineGap = 25)
        if lines is None:
            return plate
        nb_lines = len(lines);
        for line in lines:
            cv.line(edges,(line[0][0], line[0][1]),(line[0][2], line[0][3]),(255, 0 ,0))
            angle += np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0]);
        angle /= nb_lines
        x = cv.getRotationMatrix2D((plate.shape[1],plate.shape[0]), (angle * 180) / np.pi, 1)
        y = cv.warpAffine(plate, x, (plate.shape[1] + 5,plate.shape[0]), cv.INTER_CUBIC)

        return y
        
 
    def Segmentation(self, candidate):
        

            cand = cv.adaptiveThreshold(candidate,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,51,2)
            _, contours,hierarchy = cv.findContours(cand,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
            crop = []
            for c in contours:
                cnt = c
                x,y,w,h = cv.boundingRect(cnt)
                if w < (candidate.shape[1]*0.65) or h < 15:
                    continue
                crop = cand[y:y+h,x:x+w]
                break
            if len(crop) == 0:
                crop = cand
                
            crop = cv.resize(crop, (150,30))
            crop = cv.threshold(crop, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
            hProj = self.getHorizontalProjectionReduce(crop, 1)

            peaks = self.getPeaks(hProj,crop.shape[1]/30)
            x = []
            y = []
            segmentsPeaks = []
            peaks = sorted(peaks,key=operator.itemgetter(1, 0), reverse=True)
            vMax = peaks[0][1]
            for i in peaks:
                toZeroize = self.GetSegment(i, hProj, vMax)
                if toZeroize == 0:
                    break
                it = toZeroize[0]
                while it <= toZeroize[1]: 
                    hProj[it] = 0
                    it = it + 1
                    
                segmentsPeaks.append(i)
                    
                x.append(i[0])
                y.append(i[1])
            segmentsPeaks = sorted(segmentsPeaks,key=operator.itemgetter(0, 0))
            segments = []
            segIt = 0
            while True:
                if segIt + 1 >= len(segmentsPeaks):
                    break
                if segmentsPeaks[segIt+1][0] - segmentsPeaks[segIt][0] <= 7:
                    segIt = segIt + 1
                    continue
                segments.append(crop[0:crop.shape[0], segmentsPeaks[segIt][0]:segmentsPeaks[segIt+1][0]])
                segIt = segIt + 1

            #f, axarr = plt.subplots(2, sharex=True)
            #axarr[0].imshow(cv.threshold(crop, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1], 'gray')
            #axarr[1].plot(hProj)#,plt.scatter(x,y, s=10)
            #plt.savefig('segmentPlate.png')
            #input("wait...")
            #plt.show()
            chars = []
            for i in segments:
                chars.append(self.ExtractSegment(i))

            return chars
                
            

    def GetSegment(self, peak, projection, vMax):
        cw = 0.9
        if peak[1] < cw*vMax:
            return 0
        cx = 0.7
        xMax = peak[0]
        it = 1
        xl = -1
        xr = -1
        while xl == -1 or xr == -1:
            if xr == -1 and xMax + it == len(projection):
                xr = len(projection) - 1
            elif xr == -1 and projection[xMax + it] < ceil(cx*vMax):
                xr = it + xMax
            if xl == -1 and xMax - it == 0:
                xl = 0
            elif xl == -1 and projection[xMax - it] < ceil(cx*vMax):
                xl = xMax - it

            it = it + 1
        return [xl,xr]


    def ExtractSegment(self, img):
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(img, 4, cv.CV_32S)
        imgCopy = img.copy()
        heightArr = stats[:, 3]
        heightArr = heightArr[heightArr>10]
        counts = np.bincount(heightArr)
        approx = np.argmax(counts)
        possibleChars = []

        for i in range(num_labels):
            x = stats[i, cv.CC_STAT_LEFT]
            y = stats[i, cv.CC_STAT_TOP]
            w = stats[i, cv.CC_STAT_WIDTH]
            h = stats[i, cv.CC_STAT_HEIGHT]
            a = stats[i, cv.CC_STAT_AREA]

            if h < 15 or w < 5:
                continue
            roi = img[y:y+h, x: x+w]
            possibleChars.append(roi)

        return possibleChars[0]







