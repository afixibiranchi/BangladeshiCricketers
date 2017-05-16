#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 18:22:56 2017

@author: sezan92
"""

from os import listdir
from os.path import isfile,join
import numpy as np
import cv2
from skimage.feature import hog as HOG
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
import os
#Importing Models

from sklearn.svm import SVC,NuSVC

#Data Preparation

trainFolder = '/home/sezan92/BangladeshiCricketers/' 

Captain = trainFolder+'Captain'
Mehedi= trainFolder+'Mehedi'
Shabbir = trainFolder+'Sabbir'
Taskin = trainFolder+'Taskin'
Shakib = trainFolder+'Shakib'
Test = trainFolder+ 'Test'
trainData = []
responseData = []
testData = []
NumberList = []
CaptainImages = [ f for f in listdir(Captain) if isfile(join(Captain,f)) ]
MehediImages = [ f for f in listdir(Mehedi) if isfile(join(Mehedi,f)) ]
ShabbirImages = [ f for f in listdir(Shabbir) if isfile(join(Shabbir,f)) ]
TaskinImages = [ f for f in listdir(Taskin) if isfile(join(Taskin,f)) ]
ShakibImages = [ f for f in listdir(Shakib) if isfile(join(Shakib,f)) ]
TestImages = [ f for f in listdir(Test) if isfile(join(Test,f)) ]

def ReadImages(ListName,FolderName,Label):
    global NumberList
    global responseData
    global trainData
    global hog
    global cv2
    global imutils
    global winSize
    global testData
    global os
    
   
    global feature 
   
    for image in ListName:
        face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        img = cv2.imread(join(FolderName,image))
        imgray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        face = face_cascade.detectMultiScale(imgray)

        if len(face)>0:
            feature = HOG(cv2.resize(imgray[face[0][1]:face[0][1]+face[0][3],face[0][0]:face[0][0]+face[0][2]],
                                 (100,100)))
            trainData.append(feature.T)
            responseData.append(Label)
    
def num2name(num):
    if num==0:
        name = 'Captain'
    elif num==1:
        name = 'Shakib'
    elif num ==2:
        name = 'Shabbir'
    elif num ==3:
        name = 'Taskin'
    elif num ==5:
        name = 'Mehedi'
    return name


ReadImages(CaptainImages,Captain,0)
ReadImages(ShakibImages,Shakib,1)
ReadImages(ShabbirImages,Shabbir,2)
ReadImages(TaskinImages,Taskin,3)
ReadImages(MehediImages,Mehedi,5)

svm = NuSVC()
nu_options = np.arange(0.2,1)
kernel_options = ['linear','rbf']
param_grid= dict(kernel=kernel_options,nu = nu_options)
gridSVM = GridSearchCV(svm,param_grid,scoring = 'accuracy',cv=10)
X = np.float32(trainData)
y = np.float32(responseData)
gridSVM.fit(X,y)
print gridSVM.best_score_
face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
for image in TestImages:
    img = cv2.imread(join(Test,image))
    imgray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    face = face_cascade.detectMultiScale(imgray)
    if len(face)>0:
            feature = HOG(cv2.resize(imgray[face[0][1]:face[0][1]+face[0][3],face[0][0]:face[0][0]+face[0][2]],
                                 (100,100)))
    pred = gridSVM.predict(feature)
    plt.figure()
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.title(num2name(pred))
