import os
import glob
import cv2 as cv
import sys
import argparse
#from ImageProcessing import ImageProcessing
from HistImageProcessing import imageProcessing
from NeuralNetwork import NeuralNetwork
#import ComponentAnalysis
#import API

parser = argparse.ArgumentParser(description='Static ANPR system.')
parser.add_argument('-v', dest='verbose', action='store_true')
parser.add_argument('imagePath', type=str, help='Image path')
parser.set_defaults(verbose=False)
args = parser.parse_args()

# Prepare directories
chars_hist = glob.glob('./PossibleChars/hist/*')
plates = glob.glob('./PossiblePlates/*')
steps = glob.glob('./Steps/*')
bands = glob.glob('./Bands/*')
plate = glob.glob('./Plates/*')

DIRS = [ './PossibleChars', './PossiblePlates','./Steps', './PossibleChars/hist','./PossibleChars/hist/0',
         './PossibleChars/hist/1' ,'./PossibleChars/hist/2','./PossibleChars/hist/3','./PossibleChars/hist/4','./PossibleChars/hist/5','./PossibleChars/hist/6',
         './PossibleChars/hist/7','./PossibleChars/hist/8','./PossibleChars/hist/9', './Bands', './Plates']

for directory in DIRS:
    exists = os.path.isdir(directory)
    if not exists:
        os.mkdir(directory)


it = 0
while it < 10:
    chars_hist = glob.glob('./PossibleChars/hist/'+str(it)+"/*")
    it = it + 1
    for f in chars_hist:
        os.remove(f)
        
for f in plates:
    os.remove(f) 
for f in steps:
    os.remove(f)
for f in bands:
    os.remove(f) 
for f in plate:
    os.remove(f)

# loading image
imghist = imageProcessing(args.imagePath, args.verbose,(800, 600) ,5 ,9 ,20)
NN = NeuralNetwork()

it = 0

while it < 10:
    NN.predict('./PossibleChars/hist/'+str(it))
    it = 1 + it




