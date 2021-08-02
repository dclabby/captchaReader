#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:37:17 2021

@author: dclabby
"""
import os
import cv2
from utils import locateLetterRegions

# sourceFolder = '/home/dclabby/Documents/Springboard/HDAIML_SEP/Semester03/MachineLearning/Project/solving_captchas_code_examples/solving_captchas_code_examples/generated_captcha_images/'
sourceFolder = '/home/dclabby/Documents/Springboard/HDAIML_SEP/Semester03/MachineLearning/Project/code/data/separateLetters/A/'
capImages = os.listdir(sourceFolder)
capImage = capImages[1]
capLabel = capImage.split('.')[0]

imageData = cv2.imread(sourceFolder + capImage)
imageData = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)
imageData = cv2.copyMakeBorder(imageData, 8, 8, 8, 8, cv2.BORDER_REPLICATE)


from utils import resizeImage

#imageData = cv2.resize(imageData, (150, 300))
cv2.imshow(capLabel + ' - original', imageData)  
scaledImage =   resizeImage(imageData, 64, 64)
cv2.imshow(capLabel + ' - scaled', scaledImage)  
print(scaledImage.shape)
