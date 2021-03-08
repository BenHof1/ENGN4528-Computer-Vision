# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 21:01:58 2020

@author: Benjamin
"""

I = Image.open('res/images/Left.jpg')
plt.imshow()
uvt = plt.ginput(20,0)
uvt = np.array(uvt)
np.save('.../res/calibration-coordinates/uvTrans',uvt)

plt.close()
plt.imshow(I)
plt.scatter(uvt[:,0], uvt[:,1], marker='x')
J=image.open('res/images/Right.jpg')
plt.figure()
plt.imshow(J)
uv = plt.ginput(20,0)
np.save('.../res/calibration-coordinates/uv',uv)