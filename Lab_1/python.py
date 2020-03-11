# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2
import matplotlib as plt


a = np.array([[2,4,5],[5,2,200]])
b = a[0,:]
f = np.random.randn(500,1)
g=f[f<0]
x = np.zeros(100)+0.35
y=0.6*np.ones([1,len(x)])
z = x-y
a = np.linspace(1,200)
b =a[::-1]
b[b<=50]=0

img = cv2.imread('nick2.jpg',0)

shadowNick = 255 -img
cv2.imshow('shadow',shadowNick)


cv2.imshow('nick "Normal"', img)



img2=img[::-1,:]

#plt.subplot(1,1,2)
cv2.imshow('nick', img2)




img3 = (img+img2)/2
cv2.imshow('nick 3',img3)



print(img)
img4 = (np.random.randint(0,255)) +img
img5 = np.where(img4>=255,255, img4)
print(np.max(img4))
#print(img5)
cv2.imshow('nick 4',img5)
cv2.imshow('nick 5', img)

mePic = cv2.imread('face_01_u6352049.jpg')
mePic = cv2.resize(mePic, (768,512))

meRed = mePic[:,:,0]

meGreen = mePic[:,:,1]
meBlue = mePic[:,:,2]

cv2.imshow('Me 1',meRed)
cv2.imshow('Me 2',meGreen)
cv2.imshow('Me 3',meBlue)
cv2.waitKey(0)

