#
#@author: Thejus Singh Jagadish
#@date Created: 6th Feb 2018
# 
# python cluster_hist.py --dataset dataset
#


from thejus.descriptors.hist import Histogram
from sklearn.cluster import KMeans
from imutils import paths
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
ap.add_argument("-k", "--clusters", type=int, default=2)
args = vars(ap.parse_args())

desc = Histogram([8, 8, 8])
data = []

imgPath = list(paths.list_images(args["dataset"]))
imgPath = np.array(sorted(imgPath))
print(imgPath)

for path in imgPath:
	image = cv2.imread(path)
	hist = desc.describe(image)
	data.append(hist)

cluster = KMeans(n_clusters=args["clusters"])
labels = cluster.fit_predict(data)
print(labels)

for label in np.unique(labels):
	labelPath = imgPath[np.where(labels == label)]
	print(labelPath)

	for (i, path) in enumerate(labelPath):
		img = cv2.imread(path)
		cv2.imshow("Cluster {}, image #{}".format(label+1, i+1), img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()