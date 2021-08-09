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


#*****************************************************************************
from utils import resizeImage

#imageData = cv2.resize(imageData, (150, 300))
cv2.imshow(capLabel + ' - original', imageData)  
scaledImage =   resizeImage(imageData, 64, 64)
cv2.imshow(capLabel + ' - scaled', scaledImage)  
print(scaledImage.shape)


#*****************************************************************************
from keras.models import load_model

resultsFile = 'testResults~.dat'
models = ['captchaModelA.hdf5', 'captchaModelB.hdf5', 'captchaModelC.hdf5', 'captchaModelD.hdf5', 'captchaModelE.hdf5']

for modelFile in models:
    model = load_model(modelFile)
    print("\n********************\n" + modelFile + ":\n")
    model.summary()


#*****************************************************************************
import numpy as np
import matplotlib.pyplot as plt

labels = ['ModelA', 'ModelB', 'ModelC', 'ModelD', 'ModelE']
prec = [p[0] for p in prfs]
rec = [p[1] for p in prfs]
fscore = [p[2] for p in prfs]

x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/3, prec, width/3, label='Precision')
rects2 = ax.bar(x , rec, width/3, label='Recall')
rects3 = ax.bar(x + width/3, fscore, width/3, label='F score')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Loss')
ax.set_title('Presion, Recall, & F score')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower right')

fig.tight_layout()
plt.show()
plt.ylim((0.9, 1))

#*****************************************************************************
import numpy as np
import matplotlib.pyplot as plt
loss = np.array([[8.3186e-04, 0.0077],
                 [2.0940e-05, 4.3247e-04],
                 [0.0013, 0.0202],
                 [4.4992e-07, 0.0015],
                 [0.0101, 0.0187]
                 ])

labels = ['ModelA', 'ModelB', 'ModelC', 'ModelD', 'ModelE']
trainLoss = loss[:,0]
valLoss = loss[:,1]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, trainLoss, width, label='Training Loss')
rects2 = ax.bar(x + width/2, valLoss, width, label='Validation Loss')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Loss')
ax.set_title('Training & Validation Loss')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()
