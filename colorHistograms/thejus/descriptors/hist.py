#
#@author: Thejus Singh Jagadish
#@date Created: 6th Feb 2018
#

import cv2
import imutils


class Histogram:

	def __init__(self, bins):
		self.bins = bins

	def describe(self, image, mask=None):
		lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
		hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 256, 0, 256, 0, 256])
		if imutils.is_cv2():
			hist = cv2.normalize(hist)
			hist = hist.flatten()
		else:
			hist = cv2.normalise(hist, hist).flatten()

		return hist