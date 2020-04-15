# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:13:43 2020

@author: Benjamin
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

#Note: Some images will not display well using cv2.imshow(). 
#Note: Keep in mind these images are still correct and will save...
# ... to be the same images as displayed in the report

#define user functions
def my_rotation(img, angle):
    #img dimensions
    height, width = img.shape[:2]
    cx, cy = width/2, height/2
    #convert to radians and rotational references
    radians = -np.radians(angle)
    a= np.cos(radians)
    b= np.sin(radians)
    cos = abs(a)
    sin = abs(b)
    #new dimensions
    newW = abs(int((height*sin)+width*cos))
    newH = abs(int((height*cos)+width*sin))
    #kernels
    translationA_kernel = np.matrix([[1,0,-cx],[0,1,-cy],[0,0,1]])
    rot_kernel = np.matrix([[a,b,0],[-b,a,0],[0,0,1]])
    translationB_kernel = np.matrix([[1,0,newW/2],[0,1,newH/2],[0,0,1]])
    kernel = translationB_kernel*rot_kernel*translationA_kernel
    kernel = np.linalg.inv(kernel)
    output = np.zeros((newW,newH,3))
    for row in range(newH):
        for col in range(newW):
            u = int(row*kernel[0,0]+col*kernel[0,1]+kernel[0,2])
            v = int(row*kernel[1,0]+col*kernel[1,1]+kernel[1,2])
            if (u>=0 and u<width) and (v>=0 and v<height):
                pix = img[v,u]
                output[col,row] = pix
    return output;

#read in image and resize to square
mePic = cv2.imread('face_02_u6352049.jpg')
mePic =cv2.resize(mePic, (512,512))
cv2.imshow('face1', mePic)

#test each given case
ninety_anti = my_rotation(mePic, -90)
ninety = my_rotation(mePic, 90)
forty_five_anti = my_rotation(mePic, -45)
forty_five = my_rotation(mePic, 45)
fifteen_anti = my_rotation(mePic, -15)

#save images
path ='D:/Documents/UNI/CompVision/Gitlab/ENGN4528/Lab_1/Images'
cv2.imwrite(os.path.join(path, 'P6_fig24_neg_90.jpg'),ninety_anti)
cv2.imwrite(os.path.join(path, 'P6_fig24_90.jpg'),ninety)
cv2.imwrite(os.path.join(path, 'P6_fig24_45.jpg'),forty_five)
cv2.imwrite(os.path.join(path, 'P6_fig24_neg_45.jpg'),forty_five_anti)
cv2.imwrite(os.path.join(path, 'P6_fig24_neg_15.jpg'),fifteen_anti)
cv2.imwrite(os.path.join(path, 'P6_fig24_normal.jpg'),mePic)

cv2.waitKey(0)
cv2.destroyAllWindows() 