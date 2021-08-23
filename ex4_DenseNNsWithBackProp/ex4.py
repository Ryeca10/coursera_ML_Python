#############################################################
#
#  Dense one-layered NN with backprop 
#  with current setting    23/50    photos where categorized correctly
#
#############################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
import random 
import math

alpha = 0.03
lmbd = 0.001

(trainX, trainy), (testX, testy) = mnist.load_data()

inputSize = len(trainX)
n = len(trainX[0]) 

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
#  backpropagation
##################################################
def backpropStep1( theta1, theta2, a, Xi, outputLayer, yi):
    
    yVec = np.zeros([10,1])
    for i in range(10):
        if (i == yi):
            yVec[i] = 1
    
    delta3 = np.subtract(outputLayer, yVec)
    
    Delta2 = np.zeros((len(theta2), len(theta2[0])))
     
    for i in range(len(delta3)):
        for j in range(len(a)):
            Delta2[i][j] = Delta2[i][j] + a[j] * delta3[i] + lmbd * theta2[i][j]

    vecOnes = np.ones(( len(a), 1))
    
    vecOnesMinusa = np.subtract(vecOnes, a)
    aMultByOnesMinusa = np.multiply(a, vecOnesMinusa)
    theta2_T = theta2.transpose()
    thetaMultBydelta3 = np.dot(theta2_T, delta3)
    delta2 = np.multiply(thetaMultBydelta3, aMultByOnesMinusa)
    
    Delta1 = np.zeros((len(theta1), len(theta1[0])))
    for i in range(len(delta2)-1):
        for j in range(len(Xi)):
            Delta1[i][j] = Delta1[i][j] + Xi[j] * delta2[i+1] + lmbd * theta1[i][j]
            
            
    return Delta1, Delta2
    
##################################################
# train
##################################################
def train(X, y, nodeNums):
    col = np.ones((len(X),1))
    X = np.c_[X, col]
    theta1  = generateTheta(len(X[0]), nodeNums[0])
    theta2  = generateTheta(nodeNums[0]+1, nodeNums[1])
    # maxArray = np.ones((len(X),1))
    allas = np.zeros((len(X),nodeNums[0]+1))
    outputLayers = np.zeros((len(X),nodeNums[1]))
    
    
    for c in range(1500):
        D1 = np.zeros((len(theta1), len(theta1[0])))
        D2 = np.zeros((len(theta2), len(theta2[0]))) 
        for i in range(len(X)):
            a = generateLayer(X[i], theta1)
            a = np.insert(a, 0, 1., axis=0)
            aSqueezed = np.squeeze(np.asarray(a))
            outputLayer = generateLayer(aSqueezed, theta2) 
            Delta1, Delta2 = backpropStep1( theta1, theta2, a, X[i], outputLayer, y[i])
            D1 = np.add(Delta1,D1)
            D2 = np.add(Delta2,D2)
            # maximum = classification(outputLayer) 
            # maxArray[i] = maximum
        D1 = np.true_divide(D1,[len(X)])
        D2 = np.true_divide(D2,[len(X)])
        print((i+1)*(c+1))
        theta1 = theta1 - np.multiply(D1,[alpha])
        theta2 = theta2 - np.multiply(D2,[alpha])

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
    
    # backprop(theta1, theta2, allas, outputLayers, trainy)
    
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
    return theSum

print(predict(flattenTrainXShort, trainyShort, flattenTestXShort, testyShort,[25,10]))
