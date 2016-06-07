import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

MIN_MATCH_COUNT = 10

imgo = cv2.imread(sys.argv[1])
#imgo = cv2.imread('Book.png')
grayo = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create()
kpso = orb.detect(grayo, None)
kpso, deso = orb.compute(grayo, kpso)
cam = cv2.VideoCapture(0)



while True:
	ret, img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	imgall = img
	kps = orb.detect(gray, None)
	kps, des = orb.compute(gray, kps)

	#Brute Force Match with Ratio Test
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(deso, des, k = 2)
	good = []
	for m,n in matches:
		if m.distance < 0.75 * n.distance:
			good.append([m])

	# more than 10 matches, draw random color line
	if len(good)>MIN_MATCH_COUNT:
		imgall = cv2.drawMatchesKnn(imgo, kpso, img, kps, good, None, flags = 2)

	# less than 10 matches, draw black line
	else:
		draw_params = dict( matchColor = (0, 0, 0),
				singlePointColor = (0, 0, 0),
				flags = 2)
		imgall = cv2.drawMatchesKnn(imgo, kpso, img, kps, good, None, **draw_params)

	cv2.imshow('ORB match', imgall)

	if cv2.waitKey(1) == 27:
		break

cv2.destroyAllWindows()
