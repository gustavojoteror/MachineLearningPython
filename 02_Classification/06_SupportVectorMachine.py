# Support vector machine (SVM) use for classification
#
#   classifier to train your model to understand patterns within the data.
#   train         and    leter prediction
# SVM: suppervised algorithm that classifies cases by findind a separator
#       1. Mapping data to a high-dimensional feature space  (using the feature of the data)
#       2. Finding a separator (hyperplain in the higher dimension that classifies the data depending on the features).
#       3.
#   Challenging question: How do we transfer data in such a way that a separator could be drawn as a hyperplane?
#                       and How can we find the best/optimized hyperplane separator after transformation?
#
#   Transforming data
#       - Use the kernel function (mathematical function: Linear, Polynomial, Radial Basis function and Sigmoid)
# Mapping data into a higher dimensional space is called kernelling (https://en.wikipedia.org/wiki/Kernel_method)
#
#   Hyperplane separator: the idea of SVM is to find the hyperplane that best divides a dataset into two classes.
#       Option: Hyperplane the represents the largest separation (or margin) between the two classes.
#       You want to choose an hyperplane with as big a margin as possible.
#       Examples closer to the plane are support vector (data objects that are close to the hyperplane)
#       We can therefore can use only the support vector and ignore the rest of the data and find the hyperplane that as
#   the maximum distance ebtween the support vectors.
#   note: the hyperplanne and the boundary decision lines have their own equations.
#
# Summary: you use your training data to find the supporting vectors (boundary decision lines) and finding the
#       hyperplane by maximizing the margin (this is an optimization problem that is solve by gradient descent).
#
# The hyperplane: w^T*x + b = 0     W is the weight vecotr and b. Both values are find throught the optimization above.
#                                   X is the transformed data
#
#   Then use the hyperplane to find out if the new data object is above or below the hyperplane therefore classifying the data.
#
#   Pros of SVM:        Accurate in high-dimensional spaces, memory efficient (not all data use)
#   Cos of SVM:         Prone to over-fitting (features>data), No probability estimation, and Small datasets only (not computationally efficient for big datasets)
#
# Application:      Image recognition, Text category assignment, Detecting spam, Sentiment Analysis, Gene expression classification
#                   Regression, outlier detection and clustering.
#########################################################################################################
# LAB

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# The example is based on a dataset that is publicly available from the UCI Machine Learning Repository (Asuncion and Newman, 2007)[http://mlearn.ics.uci.edu/MLRepository.html].
# The dataset consists of several hundred human cell sample records, each of which contains the values of a set of cell characteristics.

# reading the data
cell_df = pd.read_csv("cell_samples.csv")
print(cell_df.shape)

# The ID field contains the patient identifiers. The characteristics of the cell samples from each patient are contained
# in fields Clump to Mit. The values are graded from 1 to 10, with 1 being the closest to benign.
# The Class field contains the diagnosis, as confirmed by separate medical procedures, as to whether the samples are
# benign (value = 2) or malignant (value = 4).
# Lets look at the distribution of the classes based on Clump thickness and Uniformity of cell size:

# plotting with pandas
ax = cell_df[cell_df['Class'] == 4][0:600].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:600].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

print(cell_df.dtypes)
#It looks like the BareNuc column includes some values that are not numerical. We can drop those rows:
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
print(cell_df.dtypes)

######################################
# let take just some features of the data
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)    # converting it into a numpy array

# now let take the classification of each object
cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])

######################################################
# Time to split the data into train and test data
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

####################################################
# Modeling with SVM in Scikit-learn)
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score


def svm_model(kernelModel,Xtrain, ytrain, Xtest, ytest):
    print('====================')
    print('Solving for kernal:', kernelModel)
    # training the model
    clf = svm.SVC(kernel=kernelModel)            # kernel can be change to 'rbf' ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    clf.fit(Xtrain, ytrain)

    # testing the model
    yhat = clf.predict(Xtest)

    from functions import plot_confusion_matrix
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(ytest, yhat, labels=[2,4])
    np.set_printoptions(precision=2)

    print (classification_report(ytest, yhat))

    # Plot non-normalized confusion matrix
    #plt.figure()
    #plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')
    #plt.show()


    print('F1 score:', f1_score(ytest, yhat, average='weighted'))
    print('jaccard index for accuracy:', jaccard_similarity_score(ytest, yhat))

svm_model('rbf',X_train, y_train, X_test, y_test)
svm_model('linear',X_train, y_train, X_test, y_test)
svm_model('poly',X_train, y_train, X_test, y_test)
svm_model('sigmoid',X_train, y_train, X_test, y_test)