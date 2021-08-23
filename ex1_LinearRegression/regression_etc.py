import numpy as np
import pandas as pd
import tensorflow as tf
# import matplotlib.pyplot as plt
# from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc


  
# normal equation    ø = (X'X)^(-1)X'y
#   loss function                    J(ø) = (1/2m)∑(i=1,m) [hø(x^(i)) - y^(i)]^2     where x^(i) is row i of X

#   gradient descent:                 øj = øj - alpha( ∂J(ø)/∂øj )
#                              =>     øj = øj - alpha( 1/m ∑(i=1,m) [hø(x^(i) - y^(i)] xj^(i) ) 


# pinv : Moore-Penrose pseudo-inverse of one or more matrices
# pinv is the psuedoinverse of a matrix that is not inversible 
#       to obtain optimal  normal equation    (X'X)^(-1)X'y
#######################################################################

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

#######################################################################
def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])
#######################################################################

A = tf.eye(5)
print(A)

dataset = pd.read_csv("ex1data1.csv")

matrix0 = dataset[dataset.columns[0]]
matrix1 = dataset[dataset.columns[1]]
# list0 = matrix0.tolist()
datalength = float(len(matrix0))



################################
## gradient descent
################################
min0 = min(matrix0)
max0 = max(matrix0)

miny = min(matrix1)
maxy = max(matrix1)


maxindex = matrix1.idxmax()
minindex = matrix1.idxmin()

minx = matrix0[minindex]
maxx = matrix0[maxindex]

theta1 = float((maxy - miny)/(maxx-minx))
theta0 = miny

tmpTheta1 = theta1 
tmpTheta0 = theta0

sigma = 0
alpha = 0.5

while (truncate(theta0,4) != truncate(tmpTheta0,4)  and  truncate(theta1,4) != truncate(tmpTheta1,4) ):

  tmpTheta1 = theta1 
  tmpTheta0 = theta0

  for i in range(datalength):
    sigma = sigma + (theta1*matrix0[i] + theta0 - matrix1[i])
  tmpTheta0 = theta0 - alpha * (1/datalength) * sigma

  for i in range(datalength):
    sigma = sigma + (theta1*matrix0[i] + theta0 - matrix1[i]) * matrix0[i]
  tmpTheta1 = theta1 - alpha * (1/datalength) * sigma





