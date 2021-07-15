# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 05:06:54 2021

@author: abc
"""


#Harris                  (it is keypoint of opencv)

import cv2
import numpy as np

img = cv2.imread("C:/Users/abc/Desktop/Digital Sreeni/Key points,detectors and descriptors in openCv/grains.jpg")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

harris = cv2.cornerHarris(gray,2,3,0.04)

img[harris>0.01*harris.max()] = [255,0,0]

cv2.imshow("Harris",img)
cv2.waitKey(0)


###################################################################################################################

#Shit-Tomasi Corner Detector & Good Feactures to Track     (it is keypoint of opencv)

import cv2
import numpy as np

img = cv2.imread("C:/Users/abc/Desktop/Digital Sreeni/Key points,detectors and descriptors in openCv/grains.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    print(x,y)
    cv2.circle(img,(x,y),3,255,-1)
    
cv2.imshow("corners",img)
cv2.waitKey(0)    


####################################################################################################################

#Fast Algorithm for corner Detection     (it is feature detector)


import cv2
import numpy as np

img = cv2.imread("C:/Users/abc/Desktop/Digital Sreeni/Key points,detectors and descriptors in openCv/grains.jpg",0)

#initialize FAST object with default values
detector = cv2.FastFeatureDetector_create(50)   #Detects 50 points

kp = detector.detect(img,None)

img2 = cv2.drawKeypoints(img,kp,None,flags=0)

cv2.imshow('Corners',img2)
cv2.waitKey(0)


##################################################################################################################

#BRIEF (Binary Robust Independent Elementary Features) 

###################################################################################################


# ORB (Oriented FAST and Rotated BRIEF)          (FAST is a detector and BRIEF is a descriptor)

import cv2
import numpy as np
img=cv2.imread("C:/Users/abc/Desktop/Digital Sreeni/Key points,detectors and descriptors in openCv/grains.jpg",0)

orb = cv2.ORB_create(50)

kp, des = orb.detectAndCompute(img,None)

img2 = cv2.drawKeypoints(img,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("ORB",img2)
cv2.waitKey(0)


####################################################################################################



                                       #THANK YOU THIS IS ALL ABOUT OPENCV
















