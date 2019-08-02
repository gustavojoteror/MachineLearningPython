# Regression is the proces sof predicting a continuos value.
#       Two types of variables: dependent   (Y) (this is the target we want to know)
#                               independent (X) (one or more: this are characteristics of the object)
#   Key point in the regressio is that our dependent value should be continuos and connot be a discrete value
#   Independent variables can be measured on either a categorical or continuos measurement scale.
# Type of regression models
#       Simple regression:  when one independent variable is used to estimated a dependent variable (Linear or non-linear)
#       Multiple regression: when several indepedent variables are used to estiamted a dependent variable (Linear or non-linear)
#
#   Application: Sales forecasing, Satisfaction Analysis, Prices Estimation, Employment income, etc....
#
# Regression Algorithms:
#       - Orginal regression
#       - Poisson regression
#       - Fast forest quantile regression
#       - Linear, Polynomial, Lasso, Stepwise, Ridge regression
#       - Bayesian linear regression
#       - Neural network regression
#       - Decision forest regression
#       - Boosted decision tree regression
#       - KNN (K-nearest neighbors)

## START: Initializing Packages
import sys                         # for sys.exit(-1)
import time                        # for timing the code
import matplotlib.pyplot as plt    # for plotting routines
import os                          # Check Existance of File
import pdb                         # Debugging Module
import numpy as np
import scipy as sp
import pandas as pd
import pylab as pl
# import sklearn as skl
from sklearn import linear_model
import math as calc
## END: Initializing Packages

