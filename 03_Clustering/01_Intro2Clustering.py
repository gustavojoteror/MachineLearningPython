# Introduction of clustering
#
# Example: customer dataset and you need to apply customer segmentation. Customer segmentation is the practice of partitioning
# a customer base into groups of individuals that have similar characteristics. Stategy that allows a business to target a
# specific group of customers so ast o more effectively allocate marketing resources.
#       One group: contain customers who are high profit and low risk (customers more likely to purchase products or subscribe)
#   Another group: contain customers from non-profit organziation.
#
# General segmentation process is not usually feasibale for large volumes of varied data
#
# An analytical approach is used to derive segments and groups from large data sets. Customers can be grouped based on
# several factors (characteristics of the object): age, gender, interests, etc.
# The idea is to divide the set of customers into categories based on characeristics they share.
#
# Clustering can group data only "unsupervised", based on the similarity of customers to each other (one of the most adopted
# approach for customers segementation). It will partition your sutomers into mutually exclusive groups. An object can be part of two groups.
#
#       Defintion of clustering:
# Clustering means finding clusters in a dataset, unsupervised.
#           A cluster is group of data points or objects in that are similar to other objects in the group (or cluster)
#           a dissimilar to data points in other clusters.
#
#       Difference between clustering and classification
#  Classification algorithms predict categorical class labels. Assigning instances to pre-defined classes. Supervised learning
#         You need a data that is already labelled to train you model and later classify a new object into the pre-defined classes
#  Clustering algorthims that cluster unlabelled data in a insupervised process. The algorithm groups similar individuals
#         depending on their shared/similar attributes
#
#       Applications
#   Retail/marketing:   identifying buying patterns of customers,
#                       recommending systems to find a group of similar iterms or users, and use it for colloborative filtering
#            Banking:   Fraud detection in credit card use
#                       Indentifying clusters of customers
#         Insurance:    Fraud detection in claims analysis
#                       Insurance risk of customers
#       Publication:    Auto-categorizing news based on their content
#                       Recommending similar news article
#          Medicine:    Characterizing patient behavior
#           Biology:    Clustering genetic markers to identify family ties
#
#   Why clustering?     Exploratory data analysis, summary generation, outlier detection (fraud or noice), finding fuplicates, pre-processing step
#
#   Clustering algorithms
#       1. Partioned-based clustering: Relatively efficient and are use for medium and large size datasets (e.g. k-means, k-Median, Fuzzy c-Means)
#       2. Hierarchical clustering: produces trees of clusters. Very intuitive and are good for small size datsets (e.g. Agglomerative, Divisive)
#       3. Density-based clustering: produces arbitrary shaped clusters. Very good for spatial clusters or when there is noise in the data (e.g. DBSCAN)