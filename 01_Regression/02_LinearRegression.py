# Linear Regression
#   y^(X1) = theta_0 + theta_1*X1   (X is independent variable and Y is the dependent variable)

## START: Initializing Packages
import matplotlib.pyplot as plt    # for plotting routines
import numpy as np
import scipy as sp
import pandas as pd
import pylab as pl
from sklearn import linear_model
## END: Initializing Packages

# Lab session one:
#   We download the data of the cars FuelConsumption.csv

# Lets read it with pandas
df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
print(df.head())    #

# Lets select some features that we want to use for regression.
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(9))

# to plot the data in terms of enginesize
# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()

#########################################
#########################################
# Model evaluation in regression model
#
#   Two types:  Train and test on the Same dataset: take the whole data set to generate the model and test it in a portion of the data
#               Train/test Split: split the data into train data (will become the data set) and the test data (unknown dependent variable)
# For this type the train and test data are not correlated.
# In both types, we have the actual values of the test data and we could predict them and check for the error.

# We will use here the train/test split evaluation
# Train/Test Split involves splitting the dataset into training and testing sets respectively,
# which are mutually exclusive. After which, you train with the training set and test with the testing set. This will provide
# a more accurate evaluation on out-of-sample accuracy because the testing dataset is not part of the dataset that have been used
# to train the data. It is more realistic for real world problems.
# This means that we know the outcome of each data point in this dataset, making it great to test with! And since this data has
# not been used to train the model, the model has no knowledge of the outcome of these data points. So, in essence, it’s truly an
# out-of-sample testing.

# let's split the data: 80% for training and 20% for testing
msk = np.random.rand(len(df)) < 0.8     # generating an array (same lenght as the data) of booleans from the condition randomNumber<0.8
print(type(msk))   # numpy.ndarray of booleans
train = cdf[msk]   # takes from the data all the true booleans
print(type(train)) # panda.dataset of booleans
test = cdf[~msk]   # takes from the data all the false booleans

print(train.head())         # this will be like the data we have
print(test.head())          # this will be like the target data we want to know off

# plotting the train and test data:
# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
# plt.scatter(test.ENGINESIZE, test.CO2EMISSIONS,  color='red')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()

###################################################################################################
###################################################################################################
###################################################################################################
# Using the regression model
#           In reality, there are multiple variables that predict the Co2emission. When more than one independent variable
# is present, the process is called multiple linear regression. For example, predicting co2emission using FUELCONSUMPTION_COMB,
# EngineSize and Cylinders of cars. The good thing here is that Multiple linear regression is the extension of simple linear regression model.

# choosing the model from scikit learn
regr = linear_model.LinearRegression()  # https://scikit-learn.org/stable/modules/linear_model.html
#   using Ordinary Least Squares: tries to estiamte the values of the coefficients by minimizing the "Mean Square Error"
#           MSE = 1/n SUM from i=1 to N (y_i - y^_i)^2 ; y^ is the approximation, y is the actual value
#           This approach uses linear algebra operations to estimate the optimal values for the coefficients
#       Downside: slow to solve the matrix. For large data sets use:
#
# Optimization Algorithm: Gradient descent (good for large data sets)
#
#
# Options in skilearn:
#   linear_model.Ridge(alpha=.5)  (imposes a penalty on the size of the coefficients. alpha>= 0 controls the amount of shrinkage )
#   linear_model.Lasso(alpha=0.1) (a linear model that estimates sparse coefficients

dependantVar = 'CO2EMISSIONS' # dependant variable, what you want to estimate
###################################################################################################
#       SIMPLE LINEAR REGRESION
# best line fit for our data
#   y^(X1) = theta_0 + theta_1*X1
# theta are the coefficients
# let first check the effect of each independent variable on the dependant variable:
indepentString = ['ENGINESIZE','CYLINDERS', 'FUELCONSUMPTION_COMB','FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']
for i in indepentString:
    x = np.asanyarray(train[[i]])
    y = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit (x, y)         #fitting the data
    # The coefficients
    print(i)
    print ('Coefficients: ', regr.coef_)    # these are the calculated coefficients
    y_hat= regr.predict(test[[i]])          # predicting the target, depedent variable, for the test data

    # but we actually know the real data
    x = np.asanyarray(test[[i]])
    y = np.asanyarray(test[['CO2EMISSIONS']])   # actual data

    print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(x, y))
    print(' ')

###################################################################################################
#       MULTI LINEAR REGRESION
# best hyperplane fit for our data
#   y^(X1,X2,X3,...) = theta_0 + theta_1*X1 + theta_2*X2 + theta_3*X3 + ...
# theta are the coefficients
# performing the regression for three data sets of fuel consumption (Combined, City and Highway)
string = ['FUELCONSUMPTION_COMB','FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']


for i in string:
    x = np.asanyarray(train[['ENGINESIZE','CYLINDERS',i]])
    y = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit (x, y)
    # The coefficients
    print(i)
    print ('Coefficients: ', regr.coef_)
    y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS',i]])
    x = np.asanyarray(test[['ENGINESIZE','CYLINDERS',i]])
    y = np.asanyarray(test[['CO2EMISSIONS']])

    print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(x, y))
    print(' ')



x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
print('FUELCONSUMPTION_CITY and FUELCONSUMPTION_HWY')
print ('Coefficients: ', regr.coef_)
y_= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"% np.mean((y_ - y) ** 2))
print('Variance score: %.2f' % regr.score(x, y))

# Q&A
#But how many independent variables should be use? They need to have a theoretical justucation, if not you will be over-fitting.
#Independet variables doesn't need to be continuos: e.g. Manual cars and automatic cars you can set in the code as Manual = 0 and Automatic = 1
# When to use linear and non-linear? Check via scatters the relationship between the independent and depedent variable.

