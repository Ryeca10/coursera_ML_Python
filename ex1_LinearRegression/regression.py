
###########################################
###########################################

# https://realpython.com/linear-regression-in-python/

###########################################

# normal equation    ø = (X'X)^(-1)X'y
#   loss function                    J(ø) = (1/2m)∑(i=1,m) [hø(x^(i)) - y^(i)]^2     where x^(i) is row i of X

#   gradient descent:                 øj = øj - alpha( ∂J(ø)/∂øj )
#                              =>     øj = øj - alpha( 1/m ∑(i=1,m) [hø(x^(i) - y^(i)] xj^(i) ) 


# pinv : Moore-Penrose pseudo-inverse of one or more matrices
# pinv is the psuedoinverse of a matrix that is not inversible 
#       to obtain optimal  normal equation    theta_hat = (X'X)^(-1)X'y
#											  y_hat = theta_hat[0] + theta_hat[1]* x
#######################################################################
###########################################
import numpy as np
# for linear regression
from sklearn.linear_model import LinearRegression
#for polynomial regression
from sklearn.preprocessing import PolynomialFeatures

###################################
#   x is just one feature
###################################

# do print(x) to understand reshape
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

model = LinearRegression()
model.fit(x, y)
# also  model = LinearRegression().fit(x, y)

###########################################
#
# linear regression and polynomial regression
#
###########################################

# coefficient of the determination
r_sq = model.score(x, y)

# linear regressions intercept b0
b0 = model.intercept_
m = model.coef_

# to predict results:
y_pred = model.predict(x)

# or :
y_pred = model.intercept_ + model.coef_ * x

# x was previously a training dataset, now lets assign x to a testing data set
# print(x_new) to see x_new
x_new = np.arange(5).reshape((-1, 1))
y_pred = model.predict(x_new)

###################################
#   Multiple Linear Regression: x represents two features
###################################

x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

model2 = LinearRegression.fit(x,y)

# coefficient of the determination
r_sq = model.score(x, y)

# linear regressions intercept b0
b0 = model2.intercept_
m = model2.coef_

# to predict results:
y_pred = model2.predict(x)

# or :
y_pred = model2.intercept_ + model2.coef_ * x


# applying to test dataset
x_new = np.arange(10).reshape((-1, 2))
y_pred = model2.predict(x_new)

###################################
#  polynomial regression of degree 2 for only one feature: features nultiple by two at each row
###################################

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])

transformer = PolynomialFeatures(degree=2, include_bias=False)
transformer.fit(x)

x_ = transformer.transform(x)
model3 = LinearRegression().fit(x_, y)

y_pred = model.predict(x_)

# another way:

x_ = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)
model3 = LinearRegression(fit_intercept=False).fit(x_, y)
y_pred = model.predict(x_)


###################################
#  polynomial regression of degree 2 for only multiplea features: features nultiple by two at each row
###################################

# Step 2a: Provide data
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

# Step 2b: Transform input data
x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

# Step 3: Create a model and fit it
model = LinearRegression().fit(x_, y)

# Step 4: Get results
r_sq = model.score(x_, y)
intercept, coefficients = model.intercept_, model.coef_

# Step 5: Predict
y_pred = model.predict(x_)
