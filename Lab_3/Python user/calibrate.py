# -*- coding: utf-8 -*-
# CLAB3
# U6352049, Benjamin Hofmann
# 8th June 2020
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from vgg_KR_from_P import *
#
#from vgg_KR_from_P.py import vgg_KR_from_P
import cv2
I = Image.open('stereo2012d.jpg')
plt.imshow(I)
#uv = plt.ginput(11,timeout=0) # Graphical user interface to get 6 points
#uv = np.array(uv)
#for use in inputing uv values. Automated using np.load
uv = np.load('values2.npy')
np.save('values2', uv)

XYZ = np.array([
    [1,0,1],[3,0,3],[1,0,3],
    [0,1,3],[0,5,3],[0,4,1],[0,1,1],
    [1,1,0],[3,1,0],[1,5,0],[3,5,0]])
#####################################################################

def calibrate(im, XYZ, uv):
    x, _ = uv.shape
    XYZ = np.hstack((XYZ, np.ones((x, 1))))
    B = np.zeros((2, 12))
    A = np.zeros((1,12))
    for i in range(x):
        B[0,:] = [0,0,0,0,-XYZ[i,0],-XYZ[i,1],-XYZ[i,2],-XYZ[i,3],uv[i,1]*XYZ[i,0],uv[i,1]*XYZ[i,1],uv[i,1]*XYZ[i,2],uv[i,1]*XYZ[i,3]]
        B[1,:] = [XYZ[i,0],XYZ[i,1],XYZ[i,2],XYZ[i,3],0,0,0,0,-uv[i,0]*XYZ[i,0],-uv[i,0]*XYZ[i,1],-uv[i,0]*XYZ[i,2],-uv[i,0]*XYZ[i,3]]
        A =np.vstack((A,B))
    A = A[1::,:]
    _,_,v =np.linalg.svd(A) #take last col, reshape
    p = np.reshape(v.T[:, -1], (3, 4))
    p = p / p[2, 3]
    return p


calibration_matrix = calibrate(I,XYZ,uv)

XYZW = np.hstack((XYZ,np.ones((11,1))))
Trig = np.array([[0,0,0,1],[1,0,0,1],[0,1,0,1],[0,0,1,1]])

test= np.matmul(calibration_matrix, XYZW.T).T
trigtest = np.matmul(calibration_matrix, Trig.T).T
for i in range(test.shape[0]):
    test[i] = test[i]/test[i,2]
for i in range(trigtest.shape[0]):
    trigtest[i] = trigtest[i]/trigtest[i,2]

plt.figure(1)
plt.imshow(I)
plt.scatter(uv[:,0],uv[:,1],marker='o', color = 'g')
plt.scatter(test[:,0],test[:,1],marker='x', color = 'b')
plt.scatter(trigtest[:,0],trigtest[:,1],marker='x', color = 'y')
plt.show()
K, R, t = vgg_KR_from_P(calibration_matrix)

print('calibration matrix is: \n', calibration_matrix)
print('K is  \n', K)
print('R is  \n', R)
print('t is  \n', t)
residual = np.sum((uv - test[:,0:2])**2)

print('The residual is:', np.sqrt(residual)/11)
'''

%% TASK 1: CALIBRATE
%
% Function to perform camera calibration
%
% Usage:   calibrate(image, XYZ, uv)
%          return C
%   Where:   image - is the image of the calibration target.
%            XYZ - is a N x 3 array of  XYZ coordinates
%                  of the calibration target points. 
%            uv  - is a N x 2 array of the image coordinates
%                  of the calibration target points.
%            K   - is the 3 x 4 camera calibration matrix.
%  The variable N should be an integer greater than or equal to 6.
%
%  This function plots the uv coordinates onto the image of the calibration
%  target. 
%
%  It also projects the XYZ coordinates back into image coordinates using
%  the calibration matrix and plots these points too as 
%  a visual check on the accuracy of the calibration process.
%
%  Lines from the origin to the vanishing points in the X, Y and Z
%  directions are overlaid on the image. 
%
%  The mean squared error between the positions of the uv coordinates 
%  and the projected XYZ coordinates is also reported.
%
%  The function should also report the error in satisfying the 
%  camera calibration matrix constraints.
% 
% your name, date 
'''

############################################################################
def homography(u2Trans, v2Trans, uBase, vBase):
    x = len(u2Trans)
    B = np.zeros((2, 9))
    A = np.zeros((1, 9))
    for i in range(x):
        B[0, :] = [-uBase[i],-vBase[i],-1,0,0,0,uBase[i]*u2Trans[i],vBase[i]*u2Trans[i],u2Trans[i]]
        B[1, :] = [0,0,0,uBase[i],vBase[i],1,-uBase[i]*v2Trans[i],-vBase[i]*v2Trans[i],-v2Trans[i]]
        A = np.vstack((A, B))
    A = A[1::, :]
    _, _, v = np.linalg.svd(A)  # = take last col, reshaped
    H = np.reshape(v.T[:, -1], (3, 3))
    H = H / H[2, 2]
    return H

'''
%% TASK 2: 
% Computes the homography H applying the Direct Linear Transformation 
% The transformation is such that 
% p = np.matmul(H, p.T), i.e.,
% (uBase, vBase, 1).T = np.matmul(H, (u2Trans , v2Trans, 1).T)
% Note: we assume (a, b, c) => np.concatenate((a, b, c), axis), be careful when 
% deal the value of axis 
%
% INPUTS: 
% u2Trans, v2Trans - vectors with coordinates u and v of the transformed image point (p') 
% uBase, vBase - vectors with coordinates u and v of the original base image point p  
% 
% OUTPUT 
% H - a 3x3 Homography matrix  
% 
% your name, date 
'''


############################################################################
def rq(A):
    # RQ factorisation

    [q,r] = np.linalg.qr(A.T)   # numpy has QR decomposition, here we can do it 
                                # with Q: orthonormal and R: upper triangle. Apply QR
                                # for the A-transpose, then A = (qr).T = r.T@q.T = RQ
    R = r.T
    Q = q.T
    return R,Q


part2 = Image.open('Left.jpg')
plt.imshow(part2)
uvt = plt.ginput(6,0)
uvt = np.array(uvt)
#uvt = np.load('calibrationleft.npy')
#np.save('calibrationleft',uvt)
#for use in inputing uv values. Automated using np.load
plt.close()
#show values just selected
plt.figure()
plt.imshow(part2)
plt.scatter(uvt[:,0], uvt[:,1], marker='o',c='r')
plt.show()

J= Image.open('Right.jpg')
plt.figure()
plt.imshow(J)
uvx = plt.ginput(6,0)
uvx = np.array(uvx)
#uvx = np.load('calibrationright.npy')
#np.save('calibrationright',uvx)
#for use in inputing uv values. Automated using np.load
#show values just selected
plt.figure()
plt.imshow(J)
plt.scatter(uvx[:,0], uvx[:,1], marker='x',c='r')
plt.show()
result = homography(uvt[:,0], uvt[:,1], uvx[:,0], uvx[:,1])
print('\n Homography Matrix is: \n', result)
im = Image.open('Left.jpg')

output = part2.transform((500,500), Image.PERSPECTIVE, result.flatten())
#warp the image using homography values in result.flatten
plt.imshow(output)
plt.show()