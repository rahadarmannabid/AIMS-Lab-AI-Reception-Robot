#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 14:32:24 2021

@author: ran
"""


import cv2
import numpy as np
import face_recognition
import os

path="database"
images=[]
classNames=[]
mylist= os.listdir(path)
print(mylist)


for clas in mylist:
    curImg=cv2.imread(f'{path}/{clas}')
    print(len(curImg))
    images.append(curImg)
    classNames.append(os.path.splitext(clas)[0])
print(classNames)

def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        #faceLoc= face_recognition.face_locations(img)[0]
        return encodeList

encodeListKnown=findEncodings(images)
print(len(encodeListKnown))
#cv2.rectangle(imageElon,(faceLoc1[3],faceLoc1[0],faceLoc1[1],faceLoc1[2]),(255,0,255), 2)