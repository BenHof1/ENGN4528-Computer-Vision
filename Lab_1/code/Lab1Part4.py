# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:12:54 2020

@author: Benjamin
"""
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

#Note: Some images will not display well using cv2.imshow(). 
#Note: Keep in mind these images are still correct and will save...
# ... to be the same images as displayed in the report

#define user functions
def color2gray(img):
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    C = 0.2989*r + 0.5870*g + 0.114*b
    return C;

def my_Gauss_filter(img, kernel):
    imgrow, imgcol = img.shape
    kernelrow, kernelcol = kernel.shape
    total =np.sum(kernel)
    output = np.zeros(img.shape)
    padx = int((kernelcol -1)/2)
    pady = int((kernelrow -1)/2)
    padimg = np.zeros((imgrow + 2*pady, imgcol + 2* padx))
    padimg[pady:padimg.shape[0]-pady, padx:padimg.shape[1]-padx] =img
    
    for row in range(imgrow):
        for col in range(imgcol):
            output[row, col] = np.sum(kernel*padimg[row:row +kernelrow, col:col + kernelcol])/total
    #replacing borders, only relevant for 5x5 kernels
    output[:,-1] = output[:,-3]
    output[:,-2] = output[:,-3]
    output[:,0] = output[:,2]
    output[:,0] = output[:,2]
    output[-1,:] = output[-3,:]
    output[-2,:] = output[-3,:]
    output[0,:] = output[2,:]
    output[1,:] = output[2,:]
    output = np.where(output>=255,255, output)
    output = np.where(output<0,0, output)
    return output;


#Read in image and display
mePic = cv2.imread('face_02_u6352049.jpg')
cv2.imshow("Starting face", mePic)

#crop, resize and set to grayscale
cropped_img = mePic[100:356, 400:656]
crop_re_size =cv2.resize(cropped_img, (256,256))
grayShape = color2gray(crop_re_size)
cv2.imshow('grey Scale', grayShape)

#establish matrix of noise values
r, c = grayShape.shape
noise = 15*np.random.randn(r,c)

#add values and clip where necessary
gaussian_noise =grayShape + noise
gaussian_noise = np.where(gaussian_noise>=255,255, gaussian_noise)
gaussian_noise = np.where(gaussian_noise<=0,0, gaussian_noise)

#Histograms of cropped image with and without noise
plt.figure(1)
plt.subplot(1,2,1)
ccounts= np.linspace(0,255,256)
histC, _ = np.histogram(crop_re_size, range=(0,255),bins=256)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.ylim(0,3000)
plt.title('Grayscale Image')
plt.bar(ccounts, histC)
plt.subplot(1,2,2)
histG, _ = np.histogram(gaussian_noise, range=(0,255),bins=256)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.title('Image with Gaussian Noise')
plt.ylim(0,3000)
plt.bar(ccounts, histG)
plt.savefig('D:/Documents/UNI/CompVision/Gitlab/ENGN4528/Lab_1/Images/P4_fig17_with_noise.png')

#steps 4 and onwards
Gauss_kernel = np.array([[1,4,6,4,1],[4,15,24,15,4],[6,24,38,24,6],[4,15,24,15,4],[1,4,6,4,1]])
filtered = my_Gauss_filter(gaussian_noise, Gauss_kernel)
cv2.imshow('filtered', filtered)

#comparison to inbuilt filter
test_official = cv2.GaussianBlur(gaussian_noise, (5,5), 1)
cv2.imshow('official', test_official)

#histogram of image after gaussian filtering
plt.figure(2)
plt.subplot(1,1,1)
histG, _ = np.histogram(filtered, range=(0,255),bins=256)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.title('Gaussian histogram')
plt.bar(ccounts, histG)
plt.savefig('D:/Documents/UNI/CompVision/Gitlab/ENGN4528/Lab_1/Images/P4_Not_used_after_filtering.png')

#saving relevant images
path ='D:/Documents/UNI/CompVision/Gitlab/ENGN4528/Lab_1/Images'
cv2.imwrite(os.path.join(path, 'P4_face1.jpg'),mePic)
cv2.imwrite(os.path.join(path, 'P4_fig15_GrayFace.jpg'),grayShape)
cv2.imwrite(os.path.join(path, 'P4_fig16_gaussian_noise.jpg'),gaussian_noise)
cv2.imwrite(os.path.join(path, 'P4_fig18_filtered_image.jpg'),filtered)
cv2.imwrite(os.path.join(path, 'P4_fig19_filter_inbuilt.jpg'),test_official)

cv2.waitKey(0)
cv2.destroyAllWindows() 