
#############################################################
#
#  logistic regression            #O(m*n + m^2*n) ≈ O(m^2*n)
#
#############################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

alpha = 0.01
#########################################
# loss function
# with sigmoid hypothesis function
# useless
#########################################
def lossFunc(X, y, theta):
	m = len(y)
	X0 = np.ones((n,1))
	Xnew = np.hstack((X,X0))
	for i in range(m):
		linear = np.dot(theta, X_new[i], out=None)
		hx = 1 / (1 + exp( (-1) * linear))
		sigmaterm = y[i] * log(hx ,2) + (1 - y[i])(log(1 - hx ,2))
		totalsum = totalsum + sigmaterm

	J = ((-1)/m) * totalsum
	return J

#####################################################
# gradient descent for logistic regression
# Xi is a vector of n elements (number of features)
# yi is a scalar
# theta us a vector of n elements
#####################################################

def gradient(X, y, j, theta): # O(m*n)
	totalsum = 0
	m = len(y)
	for i in range (m): 
		linear = np.dot(X[i], theta, out=None)   # O(n)
		totalsum = totalsum + computeStep(X[i], y[i], j, linear[0])
	return totalsum / m


def thetaupdate(Xnew, y): 
	m = len(y)
	n = len(Xnew[0])
	theta = np.zeros((n,1))
	tmptheta = np.zeros((n,1))
	count = 0
	flag = False

	for i in range (150000):           # O(1500*m*n)
		count = count + 1
		for j in range(n):				# O(m*n)
			num = alpha *  gradient(Xnew, y, j, theta)   #O(m*n)
			tmptheta[j] = theta[j] - num
		theta = tmptheta
		# print(Xnew)
		# print("********************")
		# print(theta)
	return theta


def computeStep(Xi, yi, j, linear):
	hx = 1 / (1 + math.exp( (-1) * linear))
	step = (hx - yi) * Xi[j]
	return step


##################################################
#  prediction using the updated theta vector
##################################################

def predict(X, y):
	X0 = np.ones((len(y),1))
	Xnew = np.hstack((X0,X))
	theta = thetaupdate(Xnew, y)    # O(1500*m*n)
	# print(theta)
	m = len(y)
	predict = np.zeros((m,1))
	tmppredict = np.zeros((m,1))
	for i in range (m):  		    # O(m*n)
		tmpHypothesis = np.dot(Xnew[i], theta, out=None)
		# print("*********")
		# print(Xnew[i])
		# print(theta)
		# print(tmpHypothesis)
		tmppredict[i] = 1/(1+ math.exp((-1)*tmpHypothesis))
		if tmppredict[i] >= 0.5:
			predict[i] = 1
		else:
			predict[i] = 0
	# print(tmppredict)
	return predict

##################################################
#  dataset 
##################################################

dataset = pd.read_csv("ex2data1.csv")

matrix0 = dataset[dataset.columns[0:2]].to_numpy()
matrix1 = dataset[dataset.columns[2]].to_numpy()
y_pred = predict(matrix0,matrix1)
# print(y_pred)

##################################################
#  accuracy
##################################################
sum = 0
for i in range (len(y_pred)):
	if matrix1[i] == y_pred[i]:
		sum = sum + 1

# print(y_pred.reshape(-1,1))
print(sum/len(y_pred))
