#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 09:31:20 2021

@author: dclabby
"""
import os
import cv2
import pickle
from utils import locateLetterRegions

# # Constants
# sourceFolder = '/home/dclabby/Documents/Springboard/HDAIML_SEP/Semester03/MachineLearning/Project/solving_captchas_code_examples/solving_captchas_code_examples/generated_captcha_images/'
# destFolder = './data/separateLetters'
# trainRatio = 0.8 # proportion of data set that will be used for training & validation (i.e. 1 - testRatio)

def extractLetters(sourceFolder, trainRatio=0.8, destFolder='./data/separateLetters'):
    """    

    Parameters
    ----------
    sourceFolder : string
        DESCRIPTION.
    trainRatio : float, optional
        DESCRIPTION. The default is 0.8.
    destFolder : string, optional
        DESCRIPTION. The default is './data/separateLetters'.

    Returns
    -------
    None.

    """
    
    letterCounts = {}
    
    # Get a list of all the captcha images to be processed
    capImages = os.listdir(sourceFolder)
    
    # loop over the image paths
    nImages = len(capImages) 
    # note: the original script uses all images for training (train/test split is implemented later, but test data is actually used for validation)
    # therefore, should make a train/test split here & keep the test data separate
    iSplit = int(nImages*trainRatio)
    trainTestSplit = [capImages[:iSplit], capImages[iSplit:]] # [train, test]
    
    # save the list of training and test data, so that test data can be identified later
    with open('trainTestSplit.dat', "wb") as f:
        pickle.dump(trainTestSplit, f)
    # with open('trainTestSplit.dat', "rb") as f:
    #     trainTestSplit = pickle.load(f)
    
    nTrain = len(trainTestSplit[0])
    for (iImage, capImage) in enumerate(trainTestSplit[0]):#enumerate(capImages):
        print('Processing image ' + str(iImage+1) + ' of ' + str(nTrain))#str(nImages))
        
        # Separate the filename from its extension, and use filename as the captcha's label (i.e. "2A2X.png" -> "2A2X")
        capLabel = capImage.split('.')[0]
    
        # Load image
        # imageData = cv2.imread(sourceFolder + capImage)
        imageData = cv2.imread(os.path.join(sourceFolder, capImage))
        #cv2.imshow(capLabel + ' - original', imageData)
        
        # Convert to grayscale
        imageData = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)
        #cv2.imshow(capLabel + ' - gray', imageData)    
    
        # Add padding
        imageData = cv2.copyMakeBorder(imageData, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        #cv2.imshow(capLabel + ' - padding', imageData)    
        
        # Locate letter regions
        letterRegions = locateLetterRegions(imageData)
        
        # If the number of contours does not equal the number of letters in the label it is concluded that letter extraction
        # was not successful, and this example will not be used in training data
        if len(letterRegions) != len(capLabel):
            continue
    
        # Save each letter as a separate image
        for letterRegion, letterLabel in zip(letterRegions, capLabel):
            # Get coordinates (x, y) and dimensions (w, h) of letter region
            x, y, w, h = letterRegion
            
            # extract the letter from the original image
            letterImage = imageData[y:y + h, x:x + w] 
            # # extract the letter from the original image, with a 2 pixel margin
            # letterImage = imageData[y - 2:y + h + 2, x - 2:x + w + 2] # note: image data arranged with rows corresponding to the vertical (y), & columns corresponding to the horizontal (x) 
            #cv2.imshow(letterLabel, letterImage)    
                    
            # define folder path where letters will be saved & create folder if it does not exist
            savePath = os.path.join(destFolder, letterLabel)
            if not os.path.exists(savePath):
                os.makedirs(savePath)
    
            # initialize or increment the letterCounts dictionary for the key corresponding to the present letter
            if letterLabel not in letterCounts:
                letterCounts[letterLabel] = 1
            else:
                letterCounts[letterLabel] += 1
            letterCount = letterCounts[letterLabel]
            
            # write the letter image to a file based on its letter count
            fileName = os.path.join(savePath, "{}.png".format(str(letterCount).zfill(6)))
            cv2.imwrite(fileName, letterImage)
        
