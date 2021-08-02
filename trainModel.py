#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 08:36:16 2021

@author: dclabby
"""

# from utils import generateFeaturesLabels
import pickle

# dataFolder = "./data/separateLetters"
# (features, labels) = generateFeaturesLabels(dataFolder)

# # save for upload to colab
# with open('featuresLabels.dat', "wb") as f:
#     pickle.dump((features, labels), f)

# ***** TRAINING IMPLEMENTED IN COLAB *****
# see: /content/drive/MyDrive/HD_AIML/Semester3/MachineLearning/Project/code/CaptchaBreaker.ipynb
runLocalTraining = False
if runLocalTraining:
    # import features & labels
    with open('featuresLabels.dat', "rb") as f:
        (features, labels) = pickle.load(f)

    # Split data into training & validation sets, and encode labels to one hot encodings
    from sklearn.model_selection import train_test_split
    (xTrain, xVal, yTrain, yVal) = train_test_split(features, labels, test_size=0.25, random_state=0)
    
    # Convert the labels to one-hot encodings
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer().fit(yTrain)
    yTrain = lb.transform(yTrain)
    yVal = lb.transform(yVal)
    
    # Save label encoder (use later to decode predictions)
    with open('labelEncodings.dat', "wb") as f:
        pickle.dump(lb, f)
    
    # import the necessary packages for training from keras
    from keras.models import Sequential
    from keras.layers.convolutional import Conv2D, MaxPooling2D
    from keras.layers.core import Flatten, Dense
        
    # **********
    # Define the Model A (as implemented by Geitgy)
    modelA = Sequential()
    
    # First convolutional layer with max pooling
    modelA.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
    modelA.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # Second convolutional layer with max pooling
    modelA.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
    modelA.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # Hidden layer
    modelA.add(Flatten())
    modelA.add(Dense(500, activation="relu"))
    
    # Output layer
    modelA.add(Dense(32, activation="softmax"))
    
    # compile the model
    modelA.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    # Train the neural network
    modelA.fit(xTrain, yTrain, validation_data=(xVal, yVal), batch_size=32, epochs=10, verbose=1)
    
    # **********
    # Define the Model B (sigmoid activation of output layer; binary cross entropy loss model)
    modelB = Sequential()
    
    # First convolutional layer with max pooling
    modelB.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
    modelB.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # Second convolutional layer with max pooling
    modelB.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
    modelB.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # Hidden layer
    modelB.add(Flatten())
    modelB.add(Dense(500, activation="relu"))
    
    # Output layer
    modelB.add(Dense(32, activation="sigmoid")) #see: https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
    
    # compile the model
    modelB.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]) #see: https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
    
    # Train the neural network
    modelB.fit(xTrain, yTrain, validation_data=(xVal, yVal), batch_size=32, epochs=10, verbose=1)
        
    # **********
    # Define the Model C (same as B but only one convolution/pooling layer, reduce hidden layer nodes from 500 to 250)
    modelC = Sequential()
    
    # First convolutional layer with max pooling
    modelC.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
    modelC.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # Second convolutional layer with max pooling
    #modelC.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
    #modelC.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # Hidden layer
    modelC.add(Flatten())
    #modelC.add(Dense(500, activation="relu"))
    modelC.add(Dense(250, activation="relu"))
    
    # Output layer
    modelC.add(Dense(32, activation="sigmoid")) #see: https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
    
    # compile the model
    modelC.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]) #see: https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
    
    # Train the neural network
    modelC.fit(xTrain, yTrain, validation_data=(xVal, yVal), batch_size=32, epochs=10, verbose=1)
    
    saveModels = False
    if saveModels:
        # Save the trained model to disk
        modelA.save("captchaModelA.hdf5")
        modelB.save("captchaModelB.hdf5")
        modelC.save("captchaModelC.hdf5")
    
