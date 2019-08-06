# Classification module
######  Learning objectves
## - K-neareest Neighbors
## - Decision Tree
## - SUpport Vector machines
## - Logistic Regression
## - There are more: Naive Bayes, Linear Discriminant Analysis, Neural Networks (out the scope of this course)


# Classification is a supervised learning approach. Categorizing some unknown items into a discrete set of categories or classes
# THe target attibute is a categorical variable (not a continum)
# Classification determines the class label for an unlabeled test case!
# Data===================================
#    age    address    income   debt    credit  default(label: either 1 or 0)
#     21      13          176    9.3      11.52       1    categorical value
#     30      19          46     1.3      0.52        0
#     51      41          13    3.1      19.2         1
#########################################################
#     45      21         520    4.5      52.         ??? Classification is a model that uses the data to label this object!
#
# Above is a binary classifier (only two possible labels: 0 or 1); there can be multi class classification