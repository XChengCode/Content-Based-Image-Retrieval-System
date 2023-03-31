#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
read/display images
read .cvs Created on Fri Dec  9 13:34:45 2022

@author: staff
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import csv

from extractimagefeature import ColorDescriptor, GaborDescriptor, HOGDescriptor


fig = plt.figure(figsize=(10, 7))
disp_img = np.sort(np.random.choice(9999, 100, replace=False))
i=0
j=0
for imagePath in glob.glob("images"+"/*.png"):
    # imageId = imagePath[imagePath.rfind("/")+1:]
    image = cv2.imread(imagePath)

    if j<100:
        if i==disp_img[j]:
            j+=1
            fig.add_subplot(10, 10, j)
            plt.imshow(image)
            plt.axis('off')
    else:
        break
    i+=1


#in this project we use 8 bins for hue channel,12 for saturation, 3 for value channel
bins = (8,12,3)
#initialize object
cd = ColorDescriptor(bins)

params = {"theta":4, "frequency":(0,1,0.5,0.8), "sigma":(1,3),"n_slice":2}
gd = GaborDescriptor(params)
gaborKernels = gd.kernels()

hd = HOGDescriptor()

testimage = cv2.imread("images4test/image1.png")
# plt.imshow(testimage)
HSV_hist_features_test = cd.describe(testimage)

Gabor_features_test = gd.gaborHistogram(testimage,gaborKernels)

HOG_features_test = hd.describe(testimage)

    
inputfile = open("HSV_features.csv")
reader = csv.reader(inputfile)
HSV_hist_features = np.empty((0,1440)) #feature vector length 5(segments)*8*12*3 = 1440
for row in reader:
    features = [float(x) for x in row[1:]]
    HSV_hist_features = np.append(HSV_hist_features, np.array([features]), axis=0)
inputfile.close()

inputfile = open("Gabor_features.csv")
reader = csv.reader(inputfile)
Gabor_features = np.empty((0,256)) #feature vector length = 256
for row in reader:
    features = [float(x) for x in row[1:]]
    Gabor_features = np.append(Gabor_features, np.array([features]), axis=0)
inputfile.close()

inputfile = open("HOG_features.csv")
reader = csv.reader(inputfile)
HOG_features = np.empty((0,360))
for row in reader:
    features = [float(x) for x in row[1:]]
    HOG_features = np.append(HOG_features, np.array([features]), axis=0)
inputfile.close()