## Classification module
########################
## - K-neareest Neighbors (KNN) allgorithm
# is a classification algorithm that takes a bunch of labelled points and uses them to learn how to label other points.
#
# This algorithm classifies cases based on their similarity to other cases. In k-nearest neighbors, data points that are
# near each other are said to be neighbors. KNN is based on this paradigm: "Similar cases with the same class labels ar enear each other"
# Thus the distance between two  cases is a measure of their dissimilarity.
# But how to calculate the similarity or dissimilarty between points? Example: Euclidian distance
#
# ALgorithm for k-nearest neighbors algorithm
#   1. Pick a value for K (integer):
#          low values of k can result in anomalies, this also results in overfitting taking generality of the model
#          large values of k makes the model overly generalize
#          To select the best k-value use the data itself to check the accuracy with different k values (k=1, k=2, k=3, etc..)
#          A very high value of K (ex. K = 100) produces an overly generalised model, while a very low value of k (ex. k = 1) produces a highly complex model.
#   2. Calculate the distance of unknown case from all cases
#   3. select the k-observations in the trainin data that are "nearest" to the unknwoen data points
#   4. Predict the response of the unknwon data point using the most popular response value from the k-nearest neighbors
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

# using pandas to read the dataset
df = pd.read_csv('teleCust1000t.csv')
print(df.head())   # to check the data set and the name of the columns
#region  tenure  age  marital  address  income  ed  employ  retire  gender  reside  custcat
print(df['custcat'].value_counts())  # counts the number of time each instance is in that column
#df.hist(column='income', bins=50)
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
y = df['custcat'].values
print(X[0:5])
print(y[0:5])


# Data standardization give data zero mean and unit variance, it is good practie, especially for algorithms such as KNN
# which are based on distance fo cases
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])


# let train and test the model by splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#####################
# importing knn model from sklearn
from sklearn.neighbors import KNeighborsClassifier

k = 4   # number of neighbors
#Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
print(neigh)

# predicting now
yhat = neigh.predict(X_test)
print(yhat[0:5])

#####################
# check the accuracy of the model
from sklearn import metrics
# we are using a function jaccard_similarity_score function.
# it calcualtes how closely the actual lables and predicted lavels are matched in the test set
print("Accuracy of the classifier on the Train data: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Accuracy of the classifier on the Test data:  ", metrics.accuracy_score(y_test, yhat))

# but which value of k is the best? Let's check that
Ks = 100
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))
ConfustionMx = [];
for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

print(mean_acc)

# plot model accuracy a a function of the number of neighbors
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)








