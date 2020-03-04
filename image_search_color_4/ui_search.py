import cv2
import csv
import imutils
import numpy as np
import os
import urllib.request
from flask import Flask, flash, request, redirect, render_template, url_for
from werkzeug.utils import secure_filename

########################### COLOR Descriptor Class ##########################

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



########################### Searching ###########################

filename = ''
app = Flask(__name__)

@app.route('/')
def index():
   return render_template("input.html")


@app.route('/imageupload',methods = ['POST'])
def upload_file():
    UPLOAD_FOLDER = 'queries'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    if request.method == 'POST':
        f = request.files['file']
        global filename 
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('success'))

pic = []
@app.route('/success')
def success():

    cd = FeatureExtractor((8, 12, 3))
    url = "queries/" + str(filename)
    print(url)
    query = cv2.imread(url)
    features = cd.describe(query)
    
    searcher = Searcher("index.csv")
    results = searcher.search(features)

    for (score, resultID) in results:
        pic.append("data" + "/" + str(resultID))
		
    #return 'Searched 10 files successfully.'
    return render_template('output.html', query = url, name1 = pic[0], name2 = pic[1], name3 = pic[2], 
    name4 = pic[3], name5 = pic[4], name6 = pic[5], name7 = pic[6], name8 = pic[7],
    name9 = pic[8], name10 = pic[9] )


if __name__ == '__main__':
   app.run(debug = True)



    
        

