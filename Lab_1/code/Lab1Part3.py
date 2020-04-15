# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:12:54 2020

@author: Benjamin
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

#Note: Some images will not display well using cv2.imshow(). 
#Note: Keep in mind these images are still correct and will save...
# ... to be the same images as displayed in the report

#Q3, a  Read in face, resize and display
mePic = cv2.imread('face_02_u6352049.jpg')
mePic = cv2.resize(mePic, (768,512))
cv2.imshow('original',mePic)

#Q3,b   Split into each channel
meRed = mePic[:,:,0]
meGreen = mePic[:,:,1]
meBlue = mePic[:,:,2]
cv2.imshow('Red frequencies',meRed)
cv2.imshow('Green frequencies',meGreen)
cv2.imshow('Blue frequencies',meBlue)

#histogram section
bcounts= np.linspace(0,255,256)

#histogram 1: Red
plt.figure(1)
plt.subplots_adjust(hspace=1.2, wspace=0.0)
plt.subplot(3,1,1)
hist1, _ = np.histogram(meRed, range=(0,255),bins=256)
plt.xlabel('Intensity (pixel value)')
plt.ylabel('Frequency')
plt.ylim(0,15000)
plt.title('Red-Channel histogram')
plt.bar(bcounts, hist1)
#histogram 2: Green
plt.subplot(3,1,2)
hist2, _ = np.histogram(meGreen, range=(0,255),bins=256)
plt.xlabel('Intensity (pixel value)')
plt.ylabel('Frequency')
plt.ylim(0,15000)
plt.title('Green-Channel histogram')
plt.bar(bcounts, hist2)
#histogram 3: Blue
plt.subplot(3,1,3)
hist3, _ = np.histogram(meBlue, range=(0,255),bins=256)
plt.xlabel('Intensity (pixel value)')
plt.ylabel('Frequency')
plt.ylim(0,15000)
plt.title('Blue-Channel histogram')
plt.bar(bcounts, hist3)
plt.savefig('D:/Documents/UNI/CompVision/Gitlab/ENGN4528/Lab_1/Images/P3_fig12_original_hist.png')

#Equalising Histograms
eqRed = cv2.equalizeHist(meRed)
eqGreen= cv2.equalizeHist(meGreen)
eqBlue = cv2.equalizeHist(meBlue)

#method from https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image
yuv_img = cv2.cvtColor(mePic, cv2.COLOR_BGR2YUV)
yuv_img[:,:,0] = cv2.equalizeHist(yuv_img[:,:,0])

#taking histograms of resulting equalised images
histR, _ = np.histogram(eqRed, range=(0,255),bins=256)
histG, _ = np.histogram(eqGreen, range=(0,255),bins=256)
histB, _ = np.histogram(eqBlue, range=(0,255),bins=256)
histC, _ = np.histogram(yuv_img[:,:,0], range=(0,255),bins=256)

#plotting equalised histograms
plt.figure(2)
plt.subplots_adjust(hspace=0.6, wspace=0.4)
plt.subplot(2,2,1)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.ylim(0,15000)
plt.title('Red Channel Equalised')
plt.bar(bcounts, histR)
plt.subplot(2,2,2)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.ylim(0,15000)
plt.title('Green Channel Equalised')
plt.bar(bcounts, histG)
plt.subplot(2,2,3)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.ylim(0,15000)
plt.title('Blue Channel Equalised')
plt.bar(bcounts, histB)
plt.subplot(2,2,4)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.ylim(0,15000)
plt.title('Colour (Y) Equalised')
plt.bar(bcounts, histC)
plt.savefig('D:/Documents/UNI/CompVision/Gitlab/ENGN4528/Lab_1/Images/P3_fig13_equalised_histograms.png')

#showing equalised colour image
eqColour = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
cv2.imshow('coloured image', eqColour)

#save used images
path ='D:/Documents/UNI/CompVision/Gitlab/ENGN4528/Lab_1/Images'
cv2.imwrite(os.path.join(path, 'P3_fig9_Red_Channel.jpg'),meRed)
cv2.imwrite(os.path.join(path, 'P3_fig10_Green_channel.jpg'),meGreen)
cv2.imwrite(os.path.join(path, 'P3_fig11_Blue_channel.jpg'),meBlue)
cv2.imwrite(os.path.join(path, 'P3_fig14_ Truly_Equalised_color_image.jpg'),eqColour)
cv2.imwrite(os.path.join(path, 'P3_fig8_original.jpg'),mePic)

cv2.waitKey(0)
cv2.destroyAllWindows() 

#Notes on customly equalising histograms withouth cv2.equaliseHist()
#Normalised Histograms
#cdf1 = hist1.cumsum()
#cdf_norm1 = cdf1* hist1.max()/cdf1.max()
#cdf2 = hist2.cumsum()
#cdf_norm2 = cdf2* hist2.max()/cdf2.max()
#cdf3 = hist3.cumsum()
#cdf_norm3 = cdf3* hist3.max()/cdf3.max()
#cdf =hist4.cumsum()
#cdf_norm4 = cdf4*hist4.max()/cdf.max()

#Seperate equalisations leading to distorted image
#eqTotal = np.zeros(mePic.shape)
#eqTotal[:,:,0]=eqRed
#eqTotal[:,:,1]=eqGreen
#eqTotal[:,:,2]=eqBlue
#cv2.imshow('Equalised Image', eqTotal)
#histT, _ = np.histogram(eqTotal, range=(0,255),bins=256)
#cv2.imwrite(os.path.join(path, 'P3_Equalised_color_image.jpg'),eqTotal)




