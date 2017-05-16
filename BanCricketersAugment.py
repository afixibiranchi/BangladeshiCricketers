# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 19:52:18 2016

@author: sezan1992
"""
from os import listdir
from os.path import isfile,join
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import imutils
import random
#Function

def add_noise(img):
    noise =np.random.normal(0,1,size=(img.shape))
    imNoise=img+noise
    return imNoise

def rotate_image(img):
    rotImage= imutils.rotate(img,random.uniform(0,45))
    return rotImage 
#Data Preparation
trainFolder = '/home/sezan92/BangladeshiCricketers/' 

Captain = trainFolder+'Captain'
Mehedi= trainFolder+'Mehedi'
Shabbir = trainFolder+'Sabbir'
Taskin = trainFolder+'Taskin'
Shakib = trainFolder+'Shakib'
Test = trainFolder+'Test'
FolderNames = [Captain,Mehedi,Shabbir,Taskin,Shakib]
trainData = []
responseData = []

k=0

for folder in FolderNames:
    k=k+1
    for image in listdir(folder):
        img =cv2.imread(join(folder,image))
        noisedImg =add_noise(img)
        cv2.imwrite(str(folder)+'/Noised'+str(image),noisedImg)
        noisedImg2 =add_noise(img)
        cv2.imwrite(str(folder)+'/Noised2'+str(image),noisedImg2)
        rotImage = rotate_image(img)
        cv2.imwrite(str(folder)+'/Rotated'+str(image),rotImage)
        rotImage2 = rotate_image(img)
        cv2.imwrite(str(folder)+'/Rotated2'+str(image),rotImage2)
        