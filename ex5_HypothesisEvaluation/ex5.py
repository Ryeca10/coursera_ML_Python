import scipy.io
import pandas as pd
from scipy.io import loadmat
import numpy as np
import random
import math
import matplotlib.pyplot as plt

alpha = 0.000001
lmbd = 0.0
theta = [0.0,0.0,0.0]
data_dict = loadmat('ex5data1.mat')
# print(data_dict)
X = data_dict['X']
# print(X)
y = data_dict['y']
Xval = data_dict['Xval']
yval = data_dict['yval']
Xtest = data_dict['Xtest']
ytest = data_dict['ytest']

# J([theta]) = [ 1/2m ∑i=(1,m) (h_theta(X) - y)^2 ] + [ lambda/2m ∑j=(1,n) theta_j^2]
def learningCurve(X, y, hx, theta):
    
    sumOfLoss = 0
    
    learningC = np.zeros((len(X),1))
    
    # theta[0]
    for i in range(len(X)):
        sumOfLoss = sumOfLoss + np.power((hx[i] - y[i]),2)
        learningC[i] = sumOfLoss / (i+1)
        
    
    return learningC
    
    
# ∂J/∂theta_j = [1/m ∑i=(1,m) (h_theta(X) - y)x_ij] + [lambda/m ∑j=(1,n) theta_j]
def thetaUpdate(X, y, hx, theta):
    tmptheta = theta
    
    sumTheta = 0
    sumOfLossDerivative = 0
    for i in range(len(X)):
        sumOfLossDerivative = sumOfLossDerivative + (hx[i] - y[i])
    tmptheta[0] = tmptheta[0] - (alpha/len(X)) * sumOfLossDerivative
    
    sumTheta = 0
    sumOfLossDerivative = 0
    for i in range(len(X)):
        sumOfLossDerivative = sumOfLossDerivative + (hx[i] - y[i]) * X[i][0]
    tmptheta[1] = tmptheta[1] - (alpha/len(X)) * (sumOfLossDerivative + lmbd * tmptheta[1])
    
    
    sumTheta = 0
    sumOfLossDerivative = 0
    for i in range(len(X)):
        sumOfLossDerivative = sumOfLossDerivative + (hx[i] - y[i]) * X[i][0] * X[i][0]
    tmptheta[2] = tmptheta[2] - (alpha/len(X)) * (sumOfLossDerivative + lmbd * tmptheta[2])
    
    theta = tmptheta
    return theta
              
def hx(X, theta):
    hx = np.zeros((len(X),1))
    for i in range(len(X)):
            hx[i] =  X[i][0] * X[i][0] * theta[2] + X[i][0] * theta[1] + theta[0] 
    return hx
 

for i in range(1500):
    theta = thetaUpdate(X, y, hx(X,theta), theta)

print(theta)

# hxToDraw = theta[2] * X**2 + theta[1] * X + theta[0]
plt.plot(X, hxToDraw, '-r', label=' ')
plt.title('Graph of y=2x+1')
plt.xlabel('x', color='#1C2833')
plt.ylabel('hx', color='#1C2833')
plt.legend(loc='upper left')
plt.scatter(X, y)
plt.grid()
plt.show()


J_train = learningCurve(X, y, hx(X,theta), theta)
x_axis = np.arange(len(X))

J_val = learningCurve(Xval, yval, hx(Xval,theta), theta)
x_axis_val = np.arange(len(Xval))

plt.plot(x_axis_val[2:len(x_axis_val)], J_val[2:len(J_val)], '-r', label=' ')
plt.plot(x_axis[2:len(x_axis)], J_train[2:len(J_train)], '-r', label=' ')
plt.title('Graph of y=2x+1')
plt.xlabel('num of training examples', color='#1C2833')
plt.ylabel('J', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()




