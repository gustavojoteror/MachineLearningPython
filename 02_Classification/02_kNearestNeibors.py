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
#   2. Calculate the distance of unknown case from all cases
#   3. select the k-observations in the trainin data that are "nearest" to the unknwoen data points
#   4. Predict the response of the unknwon data point using the most popular response value from the k-nearest neighbors









