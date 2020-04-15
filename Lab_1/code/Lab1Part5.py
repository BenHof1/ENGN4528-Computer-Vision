# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:13:42 2020

@author: Benjamin
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

#Note: Some images will not display well using cv2.imshow(). 
#Note: Keep in mind these images are still correct and will save...
# ... to be the same images as displayed in the report

def color2gray(img):
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    C = 0.2989*r + 0.5870*g + 0.114*b
    return C;

def my_Sobel_filter(img):
    imgrow, imgcol = img.shape
    #establish kernels used (included for simplicity rather than defining outside function)
    horizontal = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    vertical = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    kernelrow = 3
    kernelcol = 3
    h_output = np.zeros(img.shape)
    v_output = np.zeros(img.shape)
    padimg = np.zeros((imgrow + 2, imgcol + 2))
    padimg[1:padimg.shape[0]-1, 1:padimg.shape[1]-1] =img
    for row in range(imgrow):
        for col in range(imgcol):
            h_output[row, col] = np.sum(horizontal*padimg[row:row +kernelrow, col:col + kernelcol])
            v_output[row, col] = np.sum(vertical*padimg[row:row +kernelrow, col:col + kernelcol])
    #correct negative values, alternatively could take absolute values
    h_output[h_output<0] = 0
    v_output[v_output<0] = 0
    #adjust scale of values to be in correct range
    h_output = h_output*(255/h_output.max())
    v_output = v_output*(255/v_output.max())
    #clamp borders
    h_output[:,-1] = h_output[:,-2]
    h_output[:,0] = h_output[:,1]
    h_output[-1,:] = h_output[-2,:]
    h_output[0,:] = h_output[1,:]
    
    v_output[:,-1] = v_output[:,-2]
    v_output[:,0] = v_output[:,1]
    v_output[-1,:] = v_output[-2,:]
    v_output[0,:] = v_output[1,:]
    
    return h_output, v_output;

#image read in, cropped and converted to grayscale
mePic = cv2.imread('face_03_u6352049.jpg')
mePic = cv2.resize(mePic, (256,256))
grayShape = color2gray(mePic)
cv2.imshow('grayscale', grayShape)

#custom sobel filter applied
hfiltered, vfiltered = my_Sobel_filter(grayShape)
cv2.imshow('horizontal', hfiltered)
cv2.imshow('vertical', vfiltered)

#inbuilt sobel filter used
sobelx = cv2.Sobel(grayShape, -1,1,0,3)
sobely = cv2.Sobel(grayShape, -1,0,1,3)
cv2.imshow('custom sobel', sobelx)
cv2.imshow('custom sobel 2', sobely)

#images saved
path ='D:/Documents/UNI/CompVision/Gitlab/ENGN4528/Lab_1/Images'
cv2.imwrite(os.path.join(path, 'P5_Not_used_Original.jpg'),grayShape)
cv2.imwrite(os.path.join(path, 'P5_fig20_Horizontal_Sobel.jpg'),hfiltered)
cv2.imwrite(os.path.join(path, 'P5_fig22_Vertical_Sobel.jpg'),vfiltered)
cv2.imwrite(os.path.join(path, 'P5_fig23_Vertical_Sobel_inbuilt.jpg'),sobelx)
cv2.imwrite(os.path.join(path, 'P5_fig21_Horizontal_Sobel_inbuilt.jpg'),sobely)

cv2.waitKey(0)
cv2.destroyAllWindows() 
