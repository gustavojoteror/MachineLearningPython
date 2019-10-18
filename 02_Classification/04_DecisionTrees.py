# Decision trees: is to map out all possible decision paths in the form of a tree
# decision trees are about testing an attribute and branching the cases based on the result of te test.
# https://medium.com/@rishabhjain_22692/decision-trees-it-begins-here-93ff54ef134
#
#
#   Example: have the features (AGE,Sex,BloodPressure, Cholesterol) of several objects and how they responded to
#   an input: drugA and drugB. We would need to write an algorithm to decide which drug to give a new object dependent
#   on its features. So again we will use the data to generate a model (train) and use that model to predict for an unknown object (test)

#                   Age (one of the features)
#      ______________|______________
#      |             |             |
#    Young     (Middle-Age)      Senior
#      |             |             |
#     sex          drugB       Cholesterol
#   ___|___                     ___|___
#   |     |                     |     |
#   F     M                    High  Normal
#   |     |                     |     |
# drugA  drugB                 drugA drugB
#
# Each internal node coresponds to a test
# Each branch corresponds to a results of the test
# Each leaf node assigns a patient to a class
#
# Decision tree algorithm
#   1. Choose an attribute from your dataset. the one with the most predictiveness (based on decrease on impurity of nodes
#   2. Calculate the significance of attribute in splitting of data
#   3. Split data based on the value of the best attribute
#   4. Go to each branch and go back to step 1 for a difference attribute.
#
# Decision trees are built using recursive partitioning to classify the data
# The choice of attributes to split the data is very important and it is all about purity of the leaves after the split.
# A node in a tree is considered pure if, in a 100% cases the nodes fall into a specific category of a the target field.
# The impurity of nodes is calculated by entropy of data in the node. Entropy measures the randomness or uncertainty in the data.
#   For example: pure data: Node in a tree : (drugA 10, drugB 0),  Low Entropy (drugA 1, drugB 8), High Entropy (drugA 3, drugB 5)
# We look for trees with the less entropy.  (drugA 10, drugB 0) Entropy =0 ;  (drugA 3, drugB 3) Entropy=1
# Entropy of a node: frequency table of the attribute through thee entropy formula, where P is for the proportion or ratio of a category
#       Entropy= - p(A) log[p(A)] - p(B) log[p(B)]


import math
import numpy as np
drugB = 9.0  #occurances
drugA = 5.0  #occurances
numberOfCategories = 2 # drugA and drugB
total = drugA + drugB
pA = drugA/total
pB = drugB/total
entropy =- pB*math.log(pB,numberOfCategories)- pA*math.log(pA,numberOfCategories)
print(entropy)


# We select the feature which the split gave the less entropy. We choose the tree with the higher information gain after
# splitting. Information gain is the information that can increase the level of certainty after splitting.
#       InfGain = EntropyBeforeSplit - WeightedEntropyAfterSplit
# Weighted Entropy decreases => Information Gain increases
# When building a decision tree, we want to split the nodes in a way that decreases entropy and increases information gain.

################################################
#splitting by sex
# Femaale
drugB = 3.0  #occurances
drugA = 4.0  #occurances
total1 = drugA + drugB
pA = drugA/total1
pB = drugB/total1
entropy1 =- pB*math.log(pB,numberOfCategories)- pA*math.log(pA,numberOfCategories)
print(entropy1)
# Male
drugB = 6.0  #occurances
drugA = 1.0  #occurances
total2 = drugA + drugB
pA = drugA/total2
pB = drugB/total2
entropy2 =- pB*math.log(pB,numberOfCategories)- pA*math.log(pA,numberOfCategories)
print(entropy2)

infGainSex = entropy  - (total1/total)*entropy1  - (total2/total)*entropy2
print(infGainSex)

################################################
#splitting by cholesterol
# High
drugB = 3.0  #occurances
drugA = 3.0  #occurances
total1 = drugA + drugB
pA = drugA/total1
pB = drugB/total1
entropy1 =- pB*math.log(pB,numberOfCategories)- pA*math.log(pA,numberOfCategories)
print(entropy1)
# Normal
drugB = 6.0  #occurances
drugA = 2.0  #occurances
total2 = drugA + drugB
pA = drugA/total2
pB = drugB/total2
entropy2 =- pB*math.log(pB,numberOfCategories)- pA*math.log(pA,numberOfCategories)
print(entropy2)

infGainChor = entropy  - (total1/total)*entropy1  - (total2/total)*entropy2
print(infGainChor)

if(infGainChor<infGainSex):
    print('Go for sex')
else:
    print('Go for cholesterol')

print('DOOOOOOOOOOOOOOOOOING THE LAB')
########################################
# LAB
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# reading the data
my_data = pd.read_csv("drug200.csv", delimiter=",")
print(my_data[0:5])
print( "The shape of the data is", my_data.shape, " (number of object and features)")

numberCat = 5 # drugA, drugB, drugC, DrugX and Drugy

#let's split the data into features and response
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = my_data['Drug'].values


print(X[0:10])
print(y[0:5])
# As you may figure out, some features in this dataset are categorical such as Sex or BP. Unfortunately, Sklearn Decision
# Trees do not handle categorical variables. But still we can convert these features to numerical values. pandas.get_dummies()
# Convert categorical variable into dummy/indicator variables.
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

print(X[0:10])

##########################
# Setting up the Decision Tree
# We will be using train/test split on our decision tree. Let's import train_test_split from sklearn.cross_validation.
from sklearn.model_selection import train_test_split
#Now train_test_split will return 4 different parameters. We will name them:X_trainset, X_testset, y_trainset, y_testset
#The train_test_split will need the parameters: X, y, test_size=0.3, and random_state=3.
#The X and y are the arrays required before the split, the test_size represents the ratio of the testing dataset, and
# the random_state ensures that we obtain the same splits.
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

##########################
#Modeling
#We will first create an instance of the DecisionTreeClassifier called drugTree. Inside of the classifier,
# specify criterion="entropy" so we can see the information gain of each node.
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
# max_depth : int or None, optional (default=None) The maximum depth of the tree. If None, then nodes are expanded until
#                                         all leaves are pure or until all leaves contain less than min_samples_split samples.
#Next, we will fit the data with the training feature matrix X_trainset and training response vector y_trainset
drugTree.fit(X_trainset,y_trainset)
print(drugTree) # it shows the default parameters

##########################
#Prediction
# Let's make some predictions on the testing dataset and sotre it into a variable called predTree.
predTree = drugTree.predict(X_testset)
# print(predTree)

#You can print out predTree and y_testset if you want to visually compare the prediction to the actual values.
print (predTree [0:5])
print (y_testset [0:5])

##########################
# Evaluation
# Next, let's import metrics from sklearn and check the accuracy of our model.
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
# Accuracy classification score computes subset accuracy: the set of labels predicted for a sample must exactly match
# the corresponding set of labels in y_true. In multilabel classification, the function returns the subset accuracy.
# If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.


# Practice
# Can you calculate the accuracy score without sklearn ?
y_true = y_testset
sizeTest = y_true.size

print(type(y_true))
print(type(predTree))
print(y_true.size)
print(predTree.size)

count = 0
for i in range(sizeTest):
    if (predTree[i] == y_true[i]):
        count+=1

print("DecisionTrees's Accuracy without sklearn: ", count/sizeTest)

# Visualization
#Lets visualize the tree

from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')



