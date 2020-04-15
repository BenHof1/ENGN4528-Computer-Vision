# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:12:53 2020

@author: Benjamin
"""
import numpy as np
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
