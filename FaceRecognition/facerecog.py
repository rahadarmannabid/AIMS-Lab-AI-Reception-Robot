# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import face_recognition


imageElon=face_recognition.load_image_file("myface2.jpg")
imageElon=cv2.cvtColor(imageElon,cv2.COLOR_BGR2RGB)

imageJeff=face_recognition.load_image_file("myface1.jpg")
imageJeff=cv2.cvtColor(imageJeff,cv2.COLOR_BGR2RGB)

faceLoc1= face_recognition.face_locations(imageElon)[0]
encodeElon=face_recognition.face_encodings(imageElon)[0]

cv2.rectangle(imageElon,(faceLoc1[3],faceLoc1[0],faceLoc1[1],faceLoc1[2]),(255,0,255), 2)


faceLoc2= face_recognition.face_locations(imageJeff)[0]
encodeJeff=face_recognition.face_encodings(imageJeff)[0]
cv2.rectangle(imageJeff,(faceLoc2[3],faceLoc2[0],faceLoc2[1],faceLoc2[2]),(255,0,255), 2)


results=face_recognition.compare_faces([encodeElon], encodeJeff)
faceDis=face_recognition.face_distance([encodeElon], encodeJeff)
print(results)
print(faceDis)
cv2.putTest(imageJeff, f'{result} {faceDis}')
#cv2.imshow("Elon Musk", imageElon)
#cv2.imshow("Jeff", imageJeff)
#cv2.waitKey(0)