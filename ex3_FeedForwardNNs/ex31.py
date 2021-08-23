# Each pixel is represented by a floating point number indicating the grayscale intensity at that location. 
# The 28 by 28 grid of pixels is “unrolled” into a 74-dimensional vector.
# Each of these training examples becomes a single row in our data matrix X.


# Each pixel is represented by a floating point number indicating the grayscale intensity at that location. 
# The 28 by 28 grid of pixels is “unrolled” into a 74-dimensional vector.
# Each of these training examples becomes a single row in our data matrix X.


# Each pixel is represented by a floating point number indicating the grayscale intensity at that location. 
# The 28 by 28 grid of pixels is “unrolled” into a 74-dimensional vector.
# Each of these training examples becomes a single row in our data matrix X.

#############################################################
#
#  Classic FeedForward NN 
#
#############################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
import random 
import math

(trainX, trainy), (testX, testy) = mnist.load_data()

inputSize = len(trainX)
n = len(trainX[0]) 
alpha = 0.03

featureArraySize = n*n

flattenTrainX = np.zeros([inputSize, featureArraySize])
flattenTestX = np.zeros([inputSize, featureArraySize])

for i in range(inputSize):
    flattenTrainX[i] = trainX[i].flatten()
    
for i in range(len(testX)):   
    flattenTestX[i] = testX[i].flatten()
    
flattenTrainXShort = flattenTrainX[0:500,:]
flattenTestXShort = flattenTestX[0:50,:]

trainyShort = trainy[0:500]
testyShort = testy[0:50]

#####################################################
# gradient descent for logistic regression
# Xi is a vector of n elements (number of features)
# yi is a scalar
# theta us a vector of n elements
#####################################################

def thetaupdate(X, y, outputLayers, theta):
    tmptheta = theta # just to have same dimensions as theta

    sumoflossterms = 0 
    for i in range(len(theta[0])): #25
        sumoflossterms = sumoflossterms + computeStep(y[i], outputLayer[i])
        for j in range(len(theta)): #785
            tmptheta[i][j] = theta[i][j] - (alpha/len(theta[0])) *  sumoflossterms * X[i][j]
        theta = tmptheta
    return theta

def computeStep( yi, outputLayer):
    # hx = 1 / (1 + math.exp( (-1) * linear))
    totsum = 0
    
    ysub = np.zeros([len(outputLayer),1])
    
    for i in range(len(outputLayer)):
        if (i == yi):
            ysub[i] = 1
            
    for i in range(len(outputLayer)):
        totsum = totsum + (ysub[i] - outputLayer[i])

    return totsum


##################################################
#  truncating decimal points from a float value
##################################################
def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

##################################################
#  activation function
##################################################
def activationFunc(layer, theta_i):
    layer = np.squeeze(np.asarray(layer))
    theta_i = np.squeeze(np.asarray(theta_i))
    linear = np.dot(layer, theta_i, out=None)
    # linear = truncate(linear, 4)
    ai = 1 / (1 + math.exp( (-1) * linear))
    return ai

##################################################
#  hidden layer
##################################################
def generateLayer(prevLayer, theta):
    c = len(theta)
    nextLayer = np.zeros([c,1])
    for i in range(c):
        nextLayer[i] = activationFunc(prevLayer, theta[i])
    return nextLayer

##################################################
#  generate initial weights(theta)
##################################################
def generateTheta(prevLayerSize, nextLayerSize):
    tmp = nextLayerSize
    while (tmp > 1):
        tmp = tmp/10
        tmp = (-1)*tmp
    tmp = tmp/nextLayerSize
    theta = random.sample(range(nextLayerSize), nextLayerSize)
    for i in range(len(theta)):
        theta[i] = theta[i]*tmp
    for i in range(prevLayerSize - 1):
        row = random.sample(range(nextLayerSize), nextLayerSize)
        for j in range(len(row)):
            row[j] = row[j] * tmp
            tmp = (-1)*tmp
        row = np.array(row)
        col = row.reshape(-1,1)
        theta = np.c_[theta, col]
    return theta

##################################################
#  classification picks the output node with highest probability
##################################################
def classification(outputLayer):
    maximum = 0
    outputLayer = np.squeeze(np.asarray(outputLayer))
    # print("output layer is ",outputLayer)
    for i in range(len(outputLayer)):
        # print("output layer at i is ",outputLayer[i])
        # print("output layer at i+1 is ",outputLayer[i+1])
        if(outputLayer[maximum] < outputLayer[i]):
            # print("switched")
            maximum = i
    # print("maximum is ",maximum)
    return maximum
            
##################################################
# train
##################################################
def train(X, y, nodeNums):
    col = np.ones((len(X),1))
    X = np.c_[X, col]
    theta1  = generateTheta(len(X[0]), nodeNums[0])
    theta2  = generateTheta(nodeNums[0]+1, nodeNums[1])
    print("theta 1 before", theta1)
    print("theta 2 before", theta2)
    # maxArray = np.ones((len(X),1))
    allas = np.zeros((len(X),nodeNums[0]+1))
    outputLayers = np.zeros((len(X),nodeNums[1]))
    for c in range(500):
        for i in range(len(X)):
            a = generateLayer(X[i], theta1)
            a = np.insert(a, 0, 1., axis=0)
            a = np.squeeze(np.asarray(a))
            allas[i] = a
            tmpLayer = generateLayer(a, theta2) 
            tmpLayer = np.squeeze(np.asarray(tmpLayer))
            outputLayers[i] = tmpLayer
            # maximum = classification(outputLayer) 
            # maxArray[i] = maximum
        print((i+1)*(c+1))
        theta1 = thetaupdate(X, y, outputLayers, theta1)
        theta2 = thetaupdate(allas, y, outputLayers, theta2)

    return theta1, theta2

##################################################
#  prediction using the updated theta vector
##################################################
def predict(flattenTrainX, trainy, flattenTestX, testy, nodeNums):
    #feature normalization, gray scale values between 0(black) and 255(white)
    flattenTrainX = flattenTrainX/255.0
    flattenTestX = flattenTestX/255.0
    #######################
    theta1, theta2 = train(flattenTrainX, trainy, nodeNums)
    
    theSum = 0
    #for i in range (len(flattenedTestX)):  
    col = np.ones((len(flattenTestX),1))
    flattenTestX = np.c_[flattenTestX, col]
    
    #######################
    # needs correction
    #######################
    for i in range (len(testy)):
        a = generateLayer(flattenTestX[i], theta1)
        a = np.insert(a, 0, 1., axis=0)
        outputLayer = generateLayer(a, theta2)
        maximum = classification(outputLayer) 
        if testy[i] == maximum:
            theSum = theSum + 1
            
    # print(tmppredict)
    return theSum / len(testy)

print(predict(flattenTrainXShort, trainyShort, flattenTestXShort, testyShort,[25,10]))