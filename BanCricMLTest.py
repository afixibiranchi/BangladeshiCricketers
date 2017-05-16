#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 22:42:24 2017

@author: sezan92
"""
from os import listdir
from os.path import isfile,join
import numpy as np
import cv2
from skimage.feature import hog as HOG
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV

from sklearn.svm import SVC,NuSVC

trainFolder = '/home/sezan92/BangladeshiCricketers/' 
Test = trainFolder+ 'Test'
TestImages = [ f for f in listdir(Test) if isfile(join(Test,f)) ]

svm = NuSVC(kernel= 'linear',nu =0.2)