# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:06:44 2018

@author: gjoterorodrigu
"""

## START: Initializing Packages 
import sys                         # for sys.exit(-1)
import time                        # for timing the code
import matplotlib.pyplot as plt    # for plotting routines
import os                          # Check Existance of File
import pdb                         # Debugging Module
import numpy as np
import scipy as sp
import pandas as pa
import sklearn as skl
import math as calc
## END: Initializing Packages 

# Machine Learning is the subfield of computer science that gives 'computers the ability to learn without being
# explicitly programmed'
# Popular machine learning techniques:
#      1. regression/estimation:    predicting continuos values. E.g. predicting the price of a house based on its characteristics
#      2. classification:           predicting the class or category of a case. E.g. if a cell is benign or malign
#      3. clustering:               finding the strucutre of data; summarization. E.g. Can find similar patience
#      4. associations:             associating frequent co-occuring items/events. E.g. grocery items that are usyall bought together by a particular costumer.
#      5. anomaly detection:        discovering abnormal and unusual cases. E.g. for credit card fraud detection
#      6. sequence mining:          predicting next events; clcik-stream in website.
#      7. dimension reduction:      reducing the size of data (PCA)
#      8. recommendation systems:   recommending items, this associate people's preference with othere with similar tastes and recommended new items
#
#   Artificial intelligence: tries to make computers intelligent inorder to mimic the cognitive functions of humans: Computer vision, Language processing
#   Creativity and Summarization.
#
#   Machine learning is a branch of AI that covers the statistical part of artificial intelligence. It teaches the computer to solve problmes by looking at
#   hundrends of occurence and learning from them, and then using that experience to solve the same problem in new situations.
#
#   Deep learning is a special field of machine learning where computers can actually learn and makeintelligent decision on their own. This involves deeper
#   level of automation in comparison with most machine learning algorithms.
#

#   Libraries
# numpy: math library to work with n-dimensional arrays.
# scipy: collection of numerical algorithms and domain-specific toolboces, including signal processing, optimization, statistics and more.
# matplotlib: plotting package for 2D and 3D plotting.
# pandas: provides high-performance easy to use data structures. Function for data importing, manipulation and analysis.
# scikit-learn: collection of algorithms and tools for machine learning.
#               - Free software machine learning library
#               - Most of the classification, regression and clustering algorithms
#               - Works with Numpy and Scipy
#               - good documentation

# Supervised and unsupervised learning.
#   Supervised learning (uses labeled data)
#       supervised a machine learning model by 'teaching' the model: we load the model with knowldge so that we can have it predict future instances
#       We teach the model with data from labeled dataset: it is important that the data is labeled.
#       Data set you have: observations and features (of that observation). The features of the data can be given as: numerical or categorical (non-numerical)
#       Supervised learning techniques: Classification (2 above) and Regression (1 above)
#
#   Unsupervised learning (uses unlabeled data)
#       we lead the model work on its own to discover information that may not be visible for the human eye. Unsupervised algorithms trains on the dataset,
#       and draws conclusions on UNLABELED data. More complex algorithm than supervised
#       unsupervised learning techniques: Dimension reduction, Density estimation, Market basket analysis and Clustering
#                   dimension reduction (or feature selection) reduces redundant features to male the classification easier.
#                   density estimation: explore the data to find some structure within it
#                   clustering: (very popular) grouping data points or objects that are somehow similar: Discovering Structure, Summarization and Anomally detection.
#       These algorithms are less controlled
#
sys.exit()
