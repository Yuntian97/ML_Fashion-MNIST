# -*- coding: utf-8 -*-
# This part exploring deep learning CNN method with a modified VGGNet

# This is the area of Global variable
MAX_EPOCHS = 30
BATCH_SIZE = 32

# This is the area of import package
import numpy as np
import matplotlib.pyplot as plt

# import keras we need to build the CNN and Test the CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import fashion_mnist

# This function give the actual label and predict label and print out the
# summary information
def printSummary(predict, actual):
    check = (predict == actual)
    n = actual.size
    t = np.sum(check)
    percent = t/n
    print("True prediction: %d" % t)
    print("False prediction: %d" % (n - t))
    print("Correctness: " + "{:.2%}".format(percent))
    print()
    
    # return the percentage
    return percent

# This will return the keras model object of our CNN model
# This model is only used to train Fashion-MINST dataSet
def getModel():
    # Initualize the sequential model
    myModel = Sequential()
    dataShape = (28, 28, 1)
    
    # Layer conv1, we add dropout at the end of the layer to prevent overfitting
    myModel.add(Conv2D(32, (3, 3), name='conv1_1', padding="same", activation="relu", input_shape=dataShape))
    myModel.add(BatchNormalization(axis=-1))
    myModel.add(Conv2D(32, (3, 3), name='conv1_2', padding="same", activation="relu"))
    myModel.add(BatchNormalization(axis=-1))
    myModel.add(MaxPooling2D(pool_size=(2, 2)))
    myModel.add(Dropout(0.25))
    
    # layer conv2, we add dropout at the end of the layer to prevent overfitting
    myModel.add(Conv2D(64, (3, 3), name='conv2_1', padding="same", activation="relu"))
    myModel.add(BatchNormalization(axis=-1))
    myModel.add(Conv2D(64, (3, 3), name='conv2_2', padding="same", activation="relu"))
    myModel.add(BatchNormalization(axis=-1))
    myModel.add(MaxPooling2D(pool_size=(2, 2)))
    myModel.add(Dropout(0.25))
    
    # Layer conv3, we add dropout at the end of the layer to prevent overfitting
    myModel.add(Conv2D(128, (3, 3), name='conv3_1', padding="same", activation="relu"))
    myModel.add(BatchNormalization(axis=-1))
    myModel.add(Conv2D(128, (3, 3), name='conv3_2', padding="same", activation="relu"))
    myModel.add(BatchNormalization(axis=-1))
    myModel.add(MaxPooling2D(pool_size=(2, 2)))
    myModel.add(Dropout(0.25))
    
    # Layer fc4, we add dropout at the end
    myModel.add(Flatten())
    myModel.add(Dense(512, name='fc4', activation='relu'))
    myModel.add(BatchNormalization())
    myModel.add(Dropout(0.5))

    # softmax classifier, classify the results into 10 classes
    myModel.add(Dense(10, name='fc5', activation='softmax'))
    
    return myModel


# Main Script Start at here

# Section 1: Data import and data preprosessing
# We use the advantage that fashion_mnist is build in the tensorflow
((trainData, trainLabel), (testData, testLabel)) = fashion_mnist.load_data()

# Process the training and testing data, we want them to have the correct shape
# and normalize the data to range [0, 1] for faster classification
trainData = trainData.reshape((-1, 28, 28, 1)).astype(("float32")) / 255.0
testData = testData.reshape((-1, 28, 28, 1)).astype(("float32")) / 255.0

# reshape the labels to (60000, 10) and (10000, 10)
trainLabel = np.eye(10)[trainLabel]
testLabel = np.eye(10)[testLabel]

# Section 2: Get the Network Model and Training
# get the model and opt, and compile the 
model = getModel()
model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

# Start training the model
steps = model.fit(trainData, trainLabel, validation_data=(testData, testLabel), 
          batch_size=BATCH_SIZE, epochs=MAX_EPOCHS)

# Predict based on the model
predictLabel = model.predict(testData)

# Section 3: Print out the result and graph
# Use the simple function to get the predict result
printSummary(predictLabel.argmax(axis=1), testLabel.argmax(axis=1))

# Plot the graph using the steps to show what happened during each epochs
fig1 = plt.subplots()
xAxis = np.arange(0, MAX_EPOCHS)
plt.plot(xAxis, steps.history["accuracy"], label="Traning Accuracy")
plt.plot(xAxis, steps.history["val_accuracy"], label="Validation Accuracy")
plt.legend(loc="lower left")
plt.title("Network Model Performence")
plt.xlabel("Number of Epoch")
plt.ylabel("Percentage")
plt.show()

