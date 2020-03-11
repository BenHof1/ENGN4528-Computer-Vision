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
#img2=img([:-1,:])
#img3 = cv2.imshow(img2)
