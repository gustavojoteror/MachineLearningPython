# Introduction to logistic regression: used for classification
# Definition: is a statistical and machine learning technique for classifying
#           records of a dataset, based on the values of the input files
# Logistic regression is a classification algorithm for categorical variables
# indepedent variable(s) are use to predict an outcome: depedent variable
#
# Logistic regression is analagous to linear regression but tries to predict a categorical or discrete
# target field (instead of a numeric one or continous value).
# Logisitic regression we predict a binary value: Yes/No, True/False, 0/1, etc
# It can be use also for multi-class (not just 2 but more)
#
# Logistic regression: classifies an object and also gives the likelihood (probability of a case beloging
# to a specific class.
#
# WHen to use it?
#   - If the target field in your data is categorical (0/1, Yes/No, etc)
#   - If you need the probability results: logistic regression returns a probability score between 0 and 1
# for a given sample of data.
#   - If your data is linearly separable: when you need a linear decision boundary.
#    if(theta_0 + theta_1*X1 + theta_2*X2  > 0)  then Y=0
#    else                                             Y=1
# you basically generate a decision boundary
#   - if you need to understand the impact of a feature.
#   for example: if theta_1 approx 0, it means the indepedent variable X_1 as a small effect
#
# Y^ = Probabilty(y=1|x)         ,   Probabilty(y=0|x)= 1 -Probabilty(y=1|x)
#       note: y is the "labels vector" also called actual values that we would like to predict
#             y^ is the vector of the predicted values by our model.
# Main part of Logistic Regression: Sigmoid function
#
# Remember from linear regression: y^ = theta_0 + theta_1 x_1 + theta_2 x_2 + theta_3 x_3 + ...
#                                  y^ = theta^T * X ,  thetaT is called weight vector or confidence of the equation
# But in logistic regression we are classifying into a binary:
#                {  0 if thetaT * X<0.5
#           y^ = {                            we have defined a step function with a threshhold at 0.5
#                {  1 if thetaT * X>=0.5
# the only problem here is that we don't have the probability
#
# Logistic regression instead of using thetaT*X uses the Sigmoid function: sigma(thetaT*X)
#       THe Sigmoid function (instead of giving a step functions) gives a smooth distribution
#       Instead of calculating thetaT*X, it returns the probability that a thetaT*X is very big or very small
#                   Step function           Sigmoid function
#                      xxxxxxxxxxxx                 xxxxxx
#                      x                         xxx
#                      x                      xxx
#            xxxxxxxxxxx                xxxxxx
#
# The Sigmoid function (also called the logistic function). THe outcome is always between 0 and 1
#     sigma(thetaT*X) = 1 / (1 + e^(thetaT*X)) -> P(y=1|x)  which means 1 - sigma(thetaT*X) -> P(y=0|x)
#
# THe training process: selecting theta to minimize the cost function (or error) of the model.
#   1. Initialize theta with random values (standard in any machine learning model)
#   2. Calculate the model output: y^= sigma(thetaT*X) for a object
#   3. Compare the output of our model (y^) to the actual value of the object (y) and record the error(=y-y^).
#   4. Calculate the total error: the cost of the model. (note: the cost function represents how to calculate the error of the model)
#           The lower the cost the better the model. So it becomes a minimization problem.
#   5. Change theta to reduce the cost and re-do steps 2-4 until the cost is minimized.
#   There are many ways on changing theta, the most popular is using gradient descent.
#
# Cost function:
#   Cost(y^,y) =  0.5 (sigma(thetaT*X) - y)^2
#   Total cost:     J(theta) = 1/m * sum(m,i-1)Cost(y^i,yi)
# To find theta: we need to calculate the minimum point of this cost function to have the best parameters for the model
#
# However, due to the complexity it is hard to estimate the minimum to the given cost function.
#
# Simpler cost function
#               {   -log(y^)    if y=1
#   Cost(y^,y)= {
#               { -log(1-y^)    if y=0
#   Total cost      J(theta) = -1/m sum(m,i-1)[yi*log(y^i) + (1-yi)*log(1-y^i)]
#
#   To minimize the cost function we need to optimize theta. Many algorithms exist.
#   For example: gradient descent, a technique to use the derivative of a cost function to change paramter values in order
#                                   to minimize cost
#       Calculate the gradient d J(theta) / d theta1 = - 1/m sum(m,i=1)[(yi-y^)*y^i*(1-y^i)*x1]
#   the gradient vector nabla . J = [dJ/dtheta1, dJ/dtheta2, ...]
#               theta_new = theta_old - mu*(nabla . J)      note: mu is a learning rate (an additional control)
#
#   So the training algorithm
#   1. Intialize the parameters randomly
#   2. Feed the model with training set, and calculate the error/cost
#   3. Calculate the gradient of the cost function
#   4. Update theta with new parameters values using the gradients of the cost
#   5. GO to step 2 until cost is small enough
#   6. Predict the new object

###################################################3
### Lab
#  In this notebook, you will learn Logistic Regression, and then, you'll create a model for a telecommunication company,
#  to predict when its customers will leave for a competitor, so that they can take some action to retain the customers.