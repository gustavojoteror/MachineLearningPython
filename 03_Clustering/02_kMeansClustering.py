# K-means clustering: group data only "unsupervised" based on the simularity. Simplest clustering model
#
#  Types of clustering algorithms: partitioning, hierarchical or density-based clustering.
#   K-means clustering is partitioning algorithm: it divided the data into non-overlapping subsets (clusters) without
# any cluster-internal strucutre or labels (it is unsupervised algorithm)
#
# Some real-world applications of k-means:
#
    #Customer segmentation
    #Understand what the visitors of a website are trying to accomplish
    #Pattern recognition
    #Machine learning
    #Data compression
#
#
# Objects within a cluster are very similar and objects across cluster are very different or dissimilar.
#
#   key questions?
#  1. How can we find the similarity of samples in clustering?
#  2. How do we measure how similar two objects are with regard to their cluster?
#
# Instead of using similarity metric (how similar objects are) is better to us dissimilarit metrics.
# in other words: what is the distance of samples from each other is used to shape the clusters.
# k-means tries to minize the 'intra-cluster' distance and maximize the "inter-cluster" distances.
# intra-cluster: within a same cluster
# inter-cluster: between clusters
#
# k-means can calculate distance with for example euclidian distance:
# customer 1 = (x1, y1, z1)        (x,y,z are three features of the object: maybe age, income and education)
# customer 2 = (x2, y2, z2)
# dis (customer1,customer2) = sqrt(  (x1-x2)^2 + (y1-y2)^2  + (z1-z2)^2 )

# other options of distance: Euclidean distance, cosine similarity, average distance, etc..
#
# To use k-means (it is an iterative method):
# 1.Determine the number of clusters.
# The key concept of the k-Means algorithm is that it randomly picks a center point for each cluster. It means, we
# need to initialize k (number of clusters). This is not straight forward.
# 2. Generate the representative points (or centroids) of our cluster (the number of the cluster was assume in 1).
#   the representative points needs to have the number of feature as the onjects in your data set
#   how to choose the centroids? (1) Randomly choose k-observations out of the dataset or (2) Create k-observations randomly.
# 3. Assign each object to the closest centroid. For this you calculate the distance of each data point from the centroid points
#   This will generate a matrix: [numbOfObjects x k] (distance matrix). Assign the cluster to which the minimum distance is achieved.
# 4, Calcualte the error: total distance of each point from its centroid (Sum of Squares error). We need to try to reduce this error
# 5. Move the centroids: next step make the centroid the mean for data points in the cluster. Each centroid moves according
# to their cluster members.
# 6. Go back to step 3. and iterate! The algorithm stops when the centroids stop moving
#
# The iteration will results in the cluster with minimum error, or the more dense clusters.
# But it is a heuristic algorithm: there is no guarantee that it will converge to the global optimum. It can converged to
# a local optimum but maybe not to the global optimum; this depends on the inital guess.
#       TO SOLVE THIS PROBLEM: run the algorithm several times with different initial guess.
#
# Steps of k-means:
# 1.   Randomly placing k-centroids, one for each cluster. The farther apart the cluster are placed, the better.
# 2.   Calculate the distance of each point from each centroid.
# 3.   Assign each data point (object) to its closes centroid, creating a cluster
# 4.   Recalculate the position of the k-centroids. The new centrpid poistion is determined by the mean of all points in the cluster.
# 5.   Repeat seps 2-4, until the centroids no longer move.
#
#
#   ACCURACY.
#       External approach: compare the clusters with the ground truth, if it is available.
# But because k-means is an unsupervised algorithm it ussually dont have ground truth in real world problems to be used.
#       Internal approach: Average the distance between data points within a cluster.
#
# Frequent problem: determining the number of cluster in the data set. You can run the clustering across different values of k
#   and look at the metric of accuracy for clustering.
#
#   Metric of clustering: Mean distance between data points and their cluster centroid (indicates how dense our clusters are
#  or to what extend we minimzed the error of clustering)
#
#       But by increasing k we will always decrease the error: therefore we can chooce the k where the error barely reduces (elbow point)
#





##########################################################################################################################
#       LABORATORY
#we practice k-means clustering with 2 examples: (1) k-means on a random generated dataset (2) Using k-means for customer segmentation
############ (1) k-means on a random generated dataset
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

# Let first generate sample data to study clustering!
np.random.seed(0)
# making random clusters of points by using the make_blobs class. The make_blobs class can take in many inputs, but we will be using these specific ones.
#
# Input
#
#     n_samples: The total number of points equally divided among clusters.
#         Value will be: 5000
#     centers: The number of centers to generate, or the fixed center locations.
#         Value will be: [[4, 4], [-2, -1], [2, -3],[1,1]]
#     cluster_std: The standard deviation of the clusters.
#         Value will be: 0.9
#
#
# Output
#
#     X: Array of shape [n_samples, n_features]. (Feature Matrix)
#         The generated samples.
#     y: Array of shape [n_samples]. (Response Vector)
#         The integer labels for cluster membership of each sample.
#
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
# plt.scatter(X[:, 0], X[:, 1], marker='.')
# plt.show()

# Setting up K-Means
# Now that we have our random data, let's set up our K-Means Clustering.
#
# The KMeans class has many parameters that can be used, but we will be using these three:
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
#     init: Initialization method of the centroids.
#         Value will be: "k-means++"
#         k-means++: Selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
#     n_clusters: The number of clusters to form as well as the number of centroids to generate.
#         Value will be: 4 (since we have 4 centers)
#     n_init: Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the
# best output of n_init consecutive runs in terms of inertia.
#         Value will be: 12
# Initialize KMeans with these parameters, where the output parameter is called k_means.
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
# Now let's fit the KMeans model with the feature matrix we created above, X
k_means.fit(X)  # we are applying the algorithm
#Now let's grab the labels for each point in the model using KMeans' .labels_ attribute and save it as k_means_labels
k_means_labels = k_means.labels_ # this is actually the cluster assignment to each point in X
print(k_means_labels)
# We will also get the coordinates of the cluster centers using KMeans' .cluster_centers_ and save it as k_means_cluster_centers
k_means_cluster_centers = k_means.cluster_centers_  #these are the optimized centroids location
print(k_means_cluster_centers)

# Let look at it visually:

# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data poitns that are
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)

    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]

    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')

    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()

############ (2) Using k-means for customer segmentation
cust_df = pd.read_csv("Cust_Segmentation.csv")
print(cust_df.head())
# Pre-processing
# As you can see, Address in this dataset is a categorical variable. k-means algorithm isn't directly applicable to
# categorical variables because Euclidean distance function isn't really meaningful for discrete variables. So, lets drop
# this feature and run clustering.
df = cust_df.drop('Address', axis=1)
print(df.head())
# Normalizing over the standard deviation
# Now let's normalize the dataset. But why do we need normalization in the first place? Normalization is a statistical
# method that helps mathematical-based algorithms to interpret features with different magnitudes and distributions equally.
#  We use StandardScaler() to normalize our dataset.
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
print(Clus_dataSet)

# Modeling
#
# In our example (if we didn't have access to the k-means algorithm), it would be the same as guessing that each customer
# group would have certain age, income, education, etc, with multiple tests and experiments. However, using the K-means
#  clustering we can do all this process much easier.
#
# Lets apply k-means on our dataset, and take look at cluster labels.
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)
# Insights
# We assign the labels to each row in dataframe.
df["Clus_km"] = labels
print(df.head(5))
# We can easily check the centroid values by averaging the features in each cluster.
area = np.pi * ( X[:, 1])**2
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))
plt.show()

# k-means will partition your customers into mutually exclusive groups, for example, into 3 clusters. The customers in
#  each cluster are similar to each other demographically. Now we can create a profile for each group, considering the
# common characteristics of each cluster. For example, the 3 clusters can be:
#
#     AFFLUENT, EDUCATED AND OLD AGED
#     MIDDLE AGED AND MIDDLE INCOME
#     YOUNG AND LOW INCOME
#

