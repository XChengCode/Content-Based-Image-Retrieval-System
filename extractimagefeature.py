#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image feature extraction Created on Mon Dec  5 20:23:22 2022

@author: staff
"""

import numpy as np
import cv2
import imutils

from scipy import ndimage as ndi
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from skimage.feature import hog

from matplotlib import pyplot as plt

class ColorDescriptor:
	def __init__(self,bins):
		self.bins  = bins

	def describe(self,image):
		image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
		features = []

		(h,w) = image.shape[:2]
		#divide the image into 5 parts(top-left,top-right,bottom-right,bottom-left,center)
		(cx,cy) = (int(w*0.5), int(h*0.5))
		#4 corner segments
		segments = [(0,cx,0,cy),(cx,w,0,cy),(cx,w,cy,h),(0,cx,cy,h)]
		#center (ellipse shape)
		(ex,ey) = (int(w*0.75)//2, int(h*0.75)//2) #axes length
		#elliptical black mask
		ellipMask = np.zeros(image.shape[:2],dtype= "uint8")
		cv2.ellipse(ellipMask,(cx,cy),(ex,ey),0,0,360,255,-1)  # -1 :- fills entire ellipse with 255(white) color
		
		for (startX, endX, startY, endY) in segments:
			# construct a mask for each corner of the image, subtracting the elliptical center from it
			cornerMask = np.zeros(image.shape[:2],dtype = "uint8")
			cv2.rectangle(cornerMask,(startX,startY),(endX,endY),255,-1)
			cornerMask = cv2.subtract(cornerMask,ellipMask)

			hist = self.histogram(image,cornerMask)
			features.extend(hist)

		hist = self.histogram(image,ellipMask)
		features.extend(hist)

		return features


	def histogram(self,image,mask):
		hist = cv2.calcHist([image],[0,1,2],mask,self.bins,[0,180,0,256,0,256])

		if imutils.is_cv2():
			hist = cv2.normalize(hist).flatten()  #cv 2.4
		else: 
			hist = cv2.normalize(hist,hist).flatten() #cv 3+

		return hist
    

class GaborDescriptor:
	def __init__(self,params):
		self.theta = params['theta']
		self.frequency = params['frequency']
		self.sigma = params['sigma']
		self.n_slice = params['n_slice']

	def kernels(self):
		kernels = []
		for theta in range(self.theta):
			theta = theta/4. * np.pi
			for frequency in self.frequency:
				for sigma in self.sigma:
					kernel = gabor_kernel(frequency,theta=theta,sigma_x=sigma,sigma_y=sigma)
					kernels.append(kernel)
		return kernels

	def gaborHistogram(self,image,gabor_kernels):
		height,width,channel = image.shape
		#height & width of image will equally sliced into N slices
		hist = np.zeros((self.n_slice,self.n_slice,2*len(gabor_kernels))) #2*len coz to store mean and variance
		h_silce = np.around(np.linspace(0, height, self.n_slice+1, endpoint=True)).astype(int)
		w_slice = np.around(np.linspace(0, width, self.n_slice+1, endpoint=True)).astype(int)
		for hs in range(len(h_silce)-1):
			for ws in range(len(w_slice)-1):
		 		img_r = image[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
		 		hist[hs][ws] = self._gabor(img_r,gabor_kernels)

		hist /= np.sum(hist)
		#print(hist.shape)
		return hist.flatten()

	def _power(self,image,kernel):
		image = (image - image.mean()) / image.std() 
		f_img = np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 + ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
		feats = np.zeros(2, dtype=np.double)
		feats[0] = f_img.mean()
		feats[1] = f_img.var()
		return feats

	def _gabor(self,image,gabor_kernels):
		gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		results = []
		for kernel in gabor_kernels:
			results.append(self._power(gray_img,kernel))

		hist = np.array(results)
		hist = hist / np.sum(hist, axis=0)
		#print(hist.flatten())
		#print(hist.T.flatten())

		return hist.T.flatten() # .T -> transpose


class HOGDescriptor:
	def __init__(self):
		self.n_bins = 10
		self.n_slice = 6
		self.n_orient = 8
		self.pixels_per_cell = (2,2)
		self.cells_per_block = (1,1)

	def describe(self,image):
		height,width,channel = image.shape

		hist = np.zeros((self.n_slice,self.n_slice,self.n_bins)) 
		h_silce = np.around(np.linspace(0, height, self.n_slice+1, endpoint=True)).astype(int)
		w_slice = np.around(np.linspace(0, width, self.n_slice+1, endpoint=True)).astype(int)
		for hs in range(len(h_silce)-1):
			for ws in range(len(w_slice)-1):
		 		img_r = image[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
		 		hist[hs][ws] = self._HOG(img_r,self.n_bins)

		hist /= np.sum(hist)

		return hist.flatten()

	def _HOG(self,image,n_bins):
		gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		feats = hog(gray_img,orientations=self.n_orient,pixels_per_cell=self.pixels_per_cell,cells_per_block=self.cells_per_block)
		bins = np.linspace(0, np.max(feats), n_bins+1, endpoint=True)
		hist, b = np.histogram(feats,bins=bins)

		hist = np.array(hist)/np.sum(hist)

		return hist 


def chi_squared_distance(histA, histB, eps=1e-10):
    d = 0.5*np.sum([((a-b)**2)/(a+b+eps) for (a,b) in zip(histA,histB)])
    return d

def MSE_measure(features,queryFeatures):
    error = np.sum((queryFeatures - features)**2)
    return error


# class Searcher:
# 	def __init__(self,indexPath):
# 		self.indexPath = indexPath

# 	def search(self, queryFeatures, limit=10):
# 		results = {}

# 		with open(self.indexPath) as i:
# 			reader = csv.reader(i)

# 			for row in reader:
# 				features = [float(x) for x in row[1:]]
# 				d = self.chi_squared_distance(features,queryFeatures)
# 				results[row[0]] = d

# 		i.close()

# 		results = sorted([(v,k) for (k,v) in results.items()])

# 		return results[:limit]

# 	def _gsearch(self, queryFeatures, limit=10):
# 		results = {}
# 		with open(self.indexPath) as i:
# 			reader = csv.reader(i)

# 			for row in reader:
# 				features = [float(x) for x in row[1:]]
# 				error = np.sum((queryFeatures - features)**2)
# 				results[row[0]] = error

# 		i.close()

# 		results = sorted([(v,k) for (k,v) in results.items()])
# 		return results[:limit]

# 	def chi_squared_distance(self, histA, histB, eps=1e-10):

# 		d = 0.5*np.sum([((a-b)**2)/(a+b+eps) for (a,b) in zip(histA,histB)])

# 		return d


# import glob

#in this project we use 8 bins for hue channel,12 for saturation, 3 for value channel
# bins = (8,12,3)
#initialize object
# cd = ColorDescriptor(bins)

# params = {"theta":4, "frequency":(0,1,0.5,0.8), "sigma":(1,3),"n_slice":2}
# gd = GaborDescriptor(params)
# gaborKernels = gd.kernels()

# hd = HOGDescriptor()

# rows = 10
# columns = 10
# i=0

# fig = plt.figure(figsize=(10, 7))

# for imagePath in glob.glob("../"+"images"+"/*.png"):
#     imageId = imagePath[imagePath.rfind("/")+1:]
#     image = cv2.imread(imagePath)
#     HSV_hist_features = cd.describe(image)
#     Gabor_features = gd.gaborHistogram(image,gaborKernels)
#     HOG_features = hd.describe(image)
#     if i<100:
#         fig.add_subplot(rows, columns, i+1)
#         plt.imshow(image)
#         plt.axis('off')
        #plt.title("First")
    # i+=1

	#features_img = [str(f) for f in features]
	#feature vector length 5(segments)*8*12*3 = 1440
# 	output.write("%s,%s\n" % (imageId,",".join(features)))






# for imagePath in glob.glob("../../"+"database"+"/*.jpg"):
# 	imageId = imagePath[imagePath.rfind("/")+1:]
# 	image = cv2.imread(imagePath)


# 	features = [str(f) for f in features]

# 	output.write("%s,%s\n" % (imageId,",".join(features)))

# output.close()




