# Evaluation metrics for classifier
#           explain the performance of a classification model.
# using part test data we create the model.
# Later with the other part of the test data, we compare the predicted label from the model vs the actual labels from the data
#
# y: is actual labels,      y^: is predicted labels
# classifier evaluation metrics:
#   1. Jaccard Index (Jaccard Similarity coefficients)
#        J(y,y^) = |y n y^|=         |y n y^|                    |y n y^| : size of the intersection,  |y U y^| : size of the union
#                  |y U y^|    |y| + |y^| - |y n y^|
# The closer J is to 1 the better. Values closed to 0 are very bad
import numpy as np
sizee = 10
y = np.ones(sizee)
y[0:5]=0
yhat = np.ones(sizee)
yhat[2:5]=0
count =0
for i in range(sizee):
    if y[i]==yhat[i]:
        count+=1
print ('Jaccard Index ',  count/(sizee+sizee-count))
#   2. F1-score: generate confunsion matrix with the classfication results
#                                       Confusion matrix
#               chrum1       6 (true positives TP)         9 (false positive FP)         (the model calculated 15 positive from which 6 are correct)
#               chrum2       1 (false negatives FN)        24 (true negatives TN)        (the model calculated 25 negatives from which 24 are correct)
#                       y|__y^      y: true label, y^: predicted label
# Precision (measure of the accuracy) = TP /(TP+FP)   (how many where correct from the ones you guest)
# Recall  (true positive rate) = TP /(TP+FN)          (how many were correct from the ones that should have been positive)
# F1-score   = 2 / (1/Recall + 1/Precision) = 2* (Precision * Recall)/(Precision + Recall)  (the closer to 1 the F1 the better your model)
# Then we can calculate the precision and recall of each class
def f1score(TN,FN,TP,FP):
     total= TP+TN+FN+FP
     accuracy= (TP+TN)/(TP+TN+FN+FP)
     precision=TP/(TP+FP)
     recall = TP/(TP+FN)
     f1= 2/(1/precision+1/recall)
     return (f1, precision,recall,accuracy)

TN = 97750
FN = 330
FP = 150
TP = 1770
print ('total number of f1score = %f, precision = %f , recall = %f, accuracy = %f' % f1score(TN,FN,TP,FP))

TN = 25
FN = 1
FP = 9
TP = 6
print ('total number of f1score = %f, precision = %f , recall = %f, accuracy = %f' % f1score(TN,FN,TP,FP))
#   3. Log Loss: for when the classifier gives the probability to get the class label, this evaluation measures the probability for
# for each prediction. So if the labels are 0 and 1 then the results is a value between 0 and 1. So if we predict a value of 0.13 and the
# actual label is 1 then the approximation is bad and we have a high log loss.
# log loss equation: (y * log(y^) + (1-y)*log(1-y^))
# total log loss of the classifier: sum(logLoss)/n (the smaller the better)
def logloss(y, yhat):
    return (y * log(yhat) + (1-y)*log(1-yhat))

