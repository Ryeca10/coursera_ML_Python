import numpy as np
# for linear regression
from sklearn.linear_model import LinearRegression
#for polynomial regression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv("ex1data1.csv")

vector0 = dataset[dataset.columns[0]].to_numpy()
vector1 = dataset[dataset.columns[1]].to_numpy()


# first half elements of vector0 , then reshaped for transpose
x1 = vector0[0:len(vector0)//2].reshape(-1,1)

# second half elements of vector0 
x2 = vector0[len(vector0)//2 + 1:len(vector0)].reshape(-1,1)

y1 = vector1[0:len(vector1)//2]
# y2 = matrix1[len(matrix1)//2 + 1:len(matrix1)]

model = LinearRegression().fit(x1,y1)
y1_pred = model.predict(x1)

print(y1_pred.reshape(-1,1))
y2_pred = model.predict(x2)
print("****************")
print(y2_pred.reshape(-1,1))