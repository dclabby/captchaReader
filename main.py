#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 11:50:48 2021

@author: dclabby
"""

from extractLetters import extractLetters
from utils import generateFeaturesLabels
from testModel import testModel
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

sourceFolder = './data/generated_captcha_images'
lettersFolder = './data/separateLetters'
featuresLabelsFile = 'featuresLabels.dat'
featureSize = (20,20)
# featuresLabelsFile = 'featuresLabels8x8.dat'
# featureSize = (8,8)
loadResults = True
# saveResults = False
resultsFile = 'testResults~.dat'#'testResults.dat'
labels = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '2', '3', '4', '5', '6', '7', '8', '9', 'error'])

# Extract Letters
runExtractLetters = False
if runExtractLetters:
    extractLetters(sourceFolder)

# Generate Features & Labels
runGenerateFeaturesLabels = False
if runGenerateFeaturesLabels:
    (features, labels) = generateFeaturesLabels(lettersFolder,featureSize)
    if len(featuresLabelsFile) > 0: # save for upload to colab
        with open(featuresLabelsFile, "wb") as f:
            pickle.dump((features, labels), f)

# ***** TRAINING IMPLEMENTED IN COLAB *****
# see: CaptchaReader.ipynb

# Test Models
models = ['captchaModelA.hdf5', 'captchaModelB.hdf5', 'captchaModelC.hdf5', 'captchaModelD.hdf5', 'captchaModelE.hdf5']
capAccuracy = []
classReport = []
confMat = []
prfs = []
for iModel, model in enumerate(models):
    
    if loadResults:# and isfile(resultsFile):
        with open(resultsFile.replace('~', model.split('.')[0][-1]), "rb") as f:
            capLabel, capPreds = pickle.load(f)
    else:
        print('\nTesting ' + model.split('.')[0])
        capLabel, capPreds = testModel(model)
        if len(resultsFile)>0:
            with open(resultsFile.replace('~', model.split('.')[0][-1]), "wb") as f:
                pickle.dump((capLabel, capPreds), f)
                
    # Evaluate Models
    capAccuracy.append(np.sum([i==j for i, j in zip(capLabel, capPreds)])/len(capLabel))
    
    lettersTrue = np.resize([[c for c in l] for l in capLabel], (-1,1))
    lettersPred = np.resize([[c for c in l] for l in capPreds], (-1,1))
    
    confMat.append(confusion_matrix(lettersTrue, lettersPred, labels))
    prfs.append(precision_recall_fscore_support(lettersTrue, lettersPred, average='weighted'))
    classReport.append(classification_report(lettersTrue, lettersPred))
    print(classReport[-1])
    
    f, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(confMat[-1], annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.title("Confusion Matrix", fontsize=20)
    plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.95)
    ax.set_yticks(np.arange(confMat[-1].shape[0]) + 0.5, minor=False)
    ax.set_xticklabels(labels, fontsize=8, rotation=360)
    ax.set_yticklabels(labels, fontsize=8, rotation=360)
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.title('Confusion Matrix for ' + model.split('.')[0])
    plt.show()