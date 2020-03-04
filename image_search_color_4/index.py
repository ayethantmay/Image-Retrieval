import glob
import cv2
import numpy as np
import imutils
#using opencv version 4.1.2


########################### Color Descriptor Class ##########################

class FeatureExtractor:
	def __init__(self, bins):
		self.bins = bins

	def describe(self, image):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []

		(h, w) = image.shape[:2]
		(cX, cY) = (int(w * 0.5), int(h * 0.5))
		segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
			(0, cX, cY, h)]



		for (startX, endX, startY, endY) in segments:
			
			cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
			cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)

			hist = self.histogram(image, cornerMask)
			features.extend(hist)
			
		return features


	def histogram(self, image, mask):
	
		hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
		#range = [0,180,0,256,0,256] Hue value lies between 0 and 180 & Saturatlies between 0 and 256ion  and Value lies between 0 and 256.
		hist = cv2.normalize(hist, hist).flatten()
		return hist



cd = FeatureExtractor((8, 12, 3))


output = open("index.csv", "w")
for imagePath in glob.glob("data" + "/*.jpeg"):
	imageID = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
	features = cd.describe(image)
	features = [str(f) for f in features]
	output.write("%s,%s\n" % (imageID, ",".join(features)))
output.close()