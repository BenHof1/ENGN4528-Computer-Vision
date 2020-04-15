# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:12:53 2020

@author: Benjamin
"""
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

#read in image as grayscale and display
img = cv2.imread('Lenna.png',0)
cv2.imshow('P2 Lenna Grayscale', img)

#negative of image
negative = 255 -img
cv2.imshow('P2 Negative',negative)

#Flipped image
flipped=img[::-1,:]
#plt.subplot(1,1,2)
cv2.imshow('P2 Vertically_Flipped', flipped)

#average of normal and flipped
normave = (img+flipped)/2
cv2.imshow('P2 Normal_Vertical_average',normave)

#read in as colour image and swap channels
color = cv2.imread('Lenna.png')
swapped_color = color
swapped_color[:,:,0] = color[:,:,2]
swapped_color[:,:,2] = color[:,:,0]
cv2.imshow('P2 swapped colors', swapped_color)

#create matrix of random values and add to grayscale image
randomness = np.random.randint(0, 256, img.shape)
random_noise = img+randomness
#clip values greater than 255
random_noise[random_noise>255]=255
cv2.imshow('P2 Random_Noise',random_noise)

#save images
path ='D:/Documents/UNI/CompVision/Gitlab/ENGN4528/Lab_1/Images'
cv2.imwrite(os.path.join(path, 'P2_lennagrayscale.jpg'),img)
cv2.imwrite(os.path.join(path, 'P2_negative.jpg'),negative)
cv2.imwrite(os.path.join(path, 'P2_vertically flipped.jpg'),flipped)
cv2.imwrite(os.path.join(path, 'P2_swapped_colors.jpg'), swapped_color)
cv2.imwrite(os.path.join(path, 'P2_normal_vertical_average.jpg'),normave)
cv2.imwrite(os.path.join(path, 'P2_random_Noise.jpg'),random_noise)

cv2.waitKey(0)
cv2.destroyAllWindows() 