#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 14:21:48 2021

@author: dclabby
"""

import cv2
import imutils
import os
import numpy as np
from imutils import paths

def locateLetterRegions(imageData, aspectRatio=1.25, regionPadding=2):
    # threshold the image (convert it to pure black and white)
    imageThresh = cv2.threshold(imageData, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] #*************
    #cv2.imshow(capLabel + ' - threshold', imageThresh)    
    
    # find contours
    contours = cv2.findContours(imageThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    letterRegions = []
    # Loop through each contour & extract letters
    for contour in contours:
        # Get bounding rectangle for the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        
        # apply padding
        x -= regionPadding
        y -= regionPadding
        h += 2*regionPadding
        w += 2*regionPadding
    
        # Compare the width and height of the contour to detect amalgamation of >1 letter in a single contour
        if w/h > aspectRatio:
            # If the width is > 1.25 the height, then it is considered too wide to contain a single letter
            # Assume that it contains 2 letters and split the region evenly into two
            splitWidth = w//2
            letterRegions.append((x, y, splitWidth, h))
            letterRegions.append((x + splitWidth, y, splitWidth, h))
        else:
            # This is a normal letter by itself
            letterRegions.append((x, y, w, h))
            
    # Sort letterRegions in ascending order of x value, to ensure that they are read L to R
    letterRegions = sorted(letterRegions, key=lambda x: x[0])
    return letterRegions 

def resizeImage(image, targetWidth, targetHeight):
    # get image dimensions (height & width)
    (h, w) = image.shape[:2]
    
    # scale image either based on width (if w > h) or height (if h > w). 
    # Note that image is scaled to specified dimension, with unspecified dimension scaled such that aspect ratio is preserved
    # Padding to bring the unspecified dimension to the target value is calculated
    if w > h:
        image = imutils.resize(image, width=targetWidth)
        padH = (targetHeight - image.shape[0]) // 2
        padW = 0
    else:
        image = imutils.resize(image, height=targetHeight)
        padH = 0
        padW = (targetWidth - image.shape[1]) // 2
    
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (targetWidth, targetHeight)) # resize to account for rounding error
    return image


def generateFeaturesLabels(dataFolder, letterSize=(20,20)):    
    # initialize the features and labels
    features = []
    labels = []
    
    # loop over the input images
    for image_file in paths.list_images(dataFolder):
        # Load the image and convert it to grayscale
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Resize the letter so it fits in a 20x20 pixel box
        image = resizeImage(image, letterSize[0], letterSize[1])
    
        # Add a third channel dimension to the image to make Keras happy
        image = np.expand_dims(image, axis=2)
    
        # Grab the name of the letter based on the folder it was in
        label = image_file.split(os.path.sep)[-2]
    
        # Add the letter image and it's label to our training data
        features.append(image)
        labels.append(label)
    
    # scale the raw pixel intensities to the range [0, 1] (this improves training)
    features = np.array(features, dtype="float") / 255.0
    labels = np.array(labels)
    return (features, labels)
    # with open('featuresLabels.dat', "wb") as f:
    #     pickle.dump((features, labels), f)