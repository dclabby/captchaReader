#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 10:14:08 2021

@author: dclabby
"""
import pickle
import cv2
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from utils import resizeImage, locateLetterRegions
from keras.models import load_model
# from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from os.path import join
# from os.path import isfile
#from plotly import tools

# sourceFolder = '/home/dclabby/Documents/Springboard/HDAIML_SEP/Semester03/MachineLearning/Project/solving_captchas_code_examples/solving_captchas_code_examples/generated_captcha_images/'
# modelVersion = 'C'
# resultsFile = 'testResults' + modelVersion + '.dat'
# loadResults = True
# saveResults = True

def testModel(modelFile, sourceFolder = './data/generated_captcha_images', 
              labelFile='labelEncodings.dat', trainTestFile='trainTestSplit.dat'):
    
    model = load_model(modelFile)
    
    layerSizes = []
    for layer in model.layers:
        layerSizes.append(layer.get_output_at(0).get_shape().as_list())
    inputShape = layerSizes[0][1:3]
    
    # if loadResults and isfile(resultsFile):
    #     with open(resultsFile, "rb") as f:
    #         capLabel, capPreds = pickle.load(f)
    # else:
    with open(labelFile, "rb") as f:
        lb = pickle.load(f)
    
    with open(trainTestFile, "rb") as f:
        tmp = pickle.load(f)
    testImages = tmp[1]
    
    capPreds = []
    capLabel = []
    nTest = len(testImages)
    # for iTest, capImage in enumerate(testImages[:100]):
    for iTest, capImage in enumerate(testImages):
        capLabel.append(capImage.split('.')[0])# = capImage.split('.')[0]
        
        imageData = cv2.imread(join(sourceFolder, capImage))
        imageData = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)
        imageData = cv2.copyMakeBorder(imageData, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        # cv2.imshow(str(capLabel) + ' - original', imageData)
        letterRegions = locateLetterRegions(imageData)
        
        # Create an output image and a list to hold our predicted letters
        # output = cv2.merge([imageData] * 3)
        prdTmp = ''#[]
           
        if len(letterRegions) != len(capLabel[-1]):
            # del capLabel[-1]
            # continue
            capPreds.append(['error' for n in capLabel[-1]])
            continue
        
        for iLetter, letterRegion in enumerate(letterRegions):  
            # Grab the coordinates of the letter in the image
            x, y, w, h = letterRegion
    
            letterImage = imageData[y:y + h, x:x + w] 
            # Extract the letter from the original image with a 2-pixel margin around the edge
            #letterImage = imageData[y - 2:y + h + 2, x - 2:x + w + 2]
    
            # Re-size the letter image to 20x20 pixels to match training data
            # letterImage = resizeImage(letterImage, 20, 20)
            letterImage = resizeImage(letterImage, inputShape[0], inputShape[1])
            #cv2.imshow(str(capLabel[-1][iLetter]) + ' - scaled', letterImage)
    
            # Turn the single image into a 4d list of images to make Keras happy
            letterImage = np.expand_dims(letterImage, axis=2) # add a third axis
            letterImage = np.expand_dims(letterImage, axis=0) # add a dimension for the example number
    
            # Ask the neural network to make a prediction
            prediction = model.predict(letterImage)
    
            # Convert the one-hot-encoded prediction back to a normal letter
            letter = lb.inverse_transform(prediction)[0]
            prdTmp += letter#.append(letter)
            
        capPreds.append(prdTmp)
        print('Test ' + str(iTest) + ' of ' + str(nTest) + '. True label: ' + capLabel[-1] + '; predicted label: ' + capPreds[-1])
    return capLabel, capPreds
        #     # draw the prediction on the output image
        #     cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        #     cv2.putText(output, letter, (x - 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # cv2.imshow("Output", output)
        
        # if saveResults:
        #     with open(resultsFile, "wb") as f:
        #         pickle.dump((capLabel, capPreds), f)

# capAccuracy = np.sum([i==j for i, j in zip(capLabel, capPreds)])/len(capLabel)

# labels = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '2', '3', '4', '5', '6', '7', '8', '9', 'error'])
# lettersTrue = np.resize([[c for c in l] for l in capLabel], (-1,1))
# lettersPred = np.resize([[c for c in l] for l in capPreds], (-1,1))

# confMat = confusion_matrix(lettersTrue, lettersPred, labels)
# p, r, f, s = precision_recall_fscore_support(lettersTrue, lettersPred, average='weighted')
# classReport = classification_report(lettersTrue, lettersPred)
# print(classReport)

# f, ax = plt.subplots(figsize=(12, 8))
# sns.heatmap(confMat, annot=True, fmt="d", linewidths=.5, ax=ax)
# plt.title("Confusion Matrix", fontsize=20)
# plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.95)
# ax.set_yticks(np.arange(confMat.shape[0]) + 0.5, minor=False)
# ax.set_xticklabels(labels, fontsize=8, rotation=360)
# ax.set_yticklabels(labels, fontsize=8, rotation=360)
# plt.xlabel('predicted label')
# plt.ylabel('true label')
# plt.title('Confusion Matrix for Model ' + modelVersion)
# plt.show()