import cv2
import csv
import imutils
import numpy as np
import os
import urllib.request

########################### COLOR Descriptor Class ##########################

class FeatureExtractor:
	def __init__(self, bins):
		self.bins = bins

	def describe(self, image):

		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []


		(h, w) = image.shape[:2]
		(cX, cY) = (int(w * 0.5), int(h * 0.5))


		segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]

		(axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
		ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
		cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)


		for (startX, endX, startY, endY) in segments:
	
			cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
			cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
			cornerMask = cv2.subtract(cornerMask, ellipMask)

		
			hist = self.histogram(image, cornerMask)
			features.extend(hist)

	
		hist = self.histogram(image, ellipMask)
		features.extend(hist)
		return features


	def histogram(self, image, mask):

		hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
		hist = cv2.normalize(hist, hist).flatten()
		return hist


########################### Searcher Class ##########################

class Searcher:
	def __init__(self, indexPath):
		self.indexPath = indexPath

	def search(self, queryFeatures, limit = 10):
		results = {}

		with open(self.indexPath) as f:
			reader = csv.reader(f)

			for row in reader:
				features = [float(x) for x in row[1:]]
				d = self.chi2_distance(features, queryFeatures)
				results[row[0]] = d
			f.close()

		results = sorted([(v, k) for (k, v) in results.items()])
		return results[:limit]



	def chi2_distance(self, histA, histB, eps = 1e-10):
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
		return d



cd = FeatureExtractor((8, 12, 3))

query = cv2.imread('queries/apple_9.jpeg')
features = cd.describe(query)


searcher = Searcher("index.csv")
results = searcher.search(features)

cv2.imshow("Query", query)

count = 0
for (score, resultID) in results:
	result = cv2.imread("data" + "/" + resultID)
	cv2.imshow("Result", result)
	count = count+1
	cv2.waitKey(0)