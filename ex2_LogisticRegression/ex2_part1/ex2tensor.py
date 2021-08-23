
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

#		h(x) = g(theta_T * x)
#		g(z) = 1/(1 + e^(-z)) , g(-∞) = 0 , g(0) = 1/2, g(+∞) = 1

#		logistic regression  (lambda is )
#
#		J(theta) = [- 1/m ∑(1,m) yi * log(h(xi)) + (1-yi) * log(1-h(xi))]  +  lambda/(2*m)∑(1,n)theta_j^2 
#       ∂J/∂theta = 1/m ∑(1,m)(h(xi) - yi)xi    where    x0 = 1
#       theta_j = theta_j - alpha * ∂J/∂theta_j 

#		regularized logistic regression  (ridge regression) (lambda is regularization parameter)
#
#		(ridge regression)
#		J(theta) = - 1/m ∑(1,m) yi * log(h(xi)) + (1-yi) * log(1-h(xi))  +  lambda/(2*m)  ∑(1,n) theta_j^2 
#		∂J/∂theta_0 = 1/m ∑(1,m)(h(xi) - yi)  
#		∂J/∂theta_j = 1/m ∑(1,m)(h(xi) - yi)xi  +  lambda/m ∑(1,n) theta_j

#		(lasso)
#		J(theta) = - 1/m ∑(1,m) yi * log(h(xi)) + (1-yi) * log(1-h(xi))  +  lambda/(2*m)  ∑(1,n) | theta_j | 
#		∂J/∂theta_0 = 1/m ∑(1,m)(h(xi) - yi)  
#		∂J/∂theta_j = 1/m ∑(1,m)(h(xi) - yi)xi  +  some derivative of the lasso term


dataset = pd.read_csv("ex2data1.csv")

matrix0 = dataset[dataset.columns[0:2]].to_numpy().reshape(-1,2)
matrix1 = dataset[dataset.columns[2]].to_numpy()
print(matrix0)

# LogisticRegression has a l2(ridge-regression) by default
model = LogisticRegression().fit(matrix0,matrix1)
y_pred = model.predict(matrix0)
print(y_pred.reshape(-1,1))
