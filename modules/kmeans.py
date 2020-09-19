from cluster import cluster
import math
import random
import sys
import numpy as np

class kmeans(cluster):

    #Constructor for kmeans class
    def __init__(self, k=5, max_iterations=100):
        self.__k = k
        self.__max_iterations = max_iterations

    #This is the euclidean distance between x and y
    def __dist(self, x, y):
        return (math.sqrt(abs(sum([(a - b) ** 2 for a, b in zip(x, y)]))))



    def __get_cluster_hypotheses(self, X, centroids):
        cluster_hypotheses = []

        for x in X:
            distances = []

            #Get distance to each centroid
            for c in centroids:
                distances.append(self.__dist(x, c))

            #Keep the minimum distance
            cluster_hypotheses.append(distances.index(min(distances)))
        return cluster_hypotheses

    
    def __get_potential_clusters(self, X, cluster_hypotheses):
        potential_clusters = {}

        #hyp is the cluster hypothesis
        #x is the data point corresponding to that hypothesis
        for hyp, x in zip(cluster_hypotheses, X):
                if hyp not in potential_clusters:
                    potential_clusters[hyp] = [x]
                else:
                    potential_clusters[hyp].append(x)

        return potential_clusters

    
    def fit(self, X):
        #Picking centroids randomly in the data
        centroids = [np.random.normal(np.mean(X, axis=0)) for x in range(self.__k)]

        #We will run the algorithm the maximum amount of times specified
        for w in range(self.__max_iterations):

            #Get the cluster hypotheses for current centroids
            cluster_hypotheses = self.__get_cluster_hypotheses(X, centroids)
            
            #Separating ech point into it's cluster to calculate new centroids
            potential_clusters = self.__get_potential_clusters(X, cluster_hypotheses)

            #Taking the mean of each cluster and assigning it to its centroid
            for key in potential_clusters:
                centroids[key] = np.mean(potential_clusters[key], axis=0)

        return([cluster_hypotheses] + [centroids])