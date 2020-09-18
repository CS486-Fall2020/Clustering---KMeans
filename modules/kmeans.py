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
    def dist(self, x, y):
        return (math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)])))

        
    def fit(self, X):

        #Picking centroids randomly in the data
        centroids = list(np.array(random.choices(X, k=self.__k)))

        #We will run the algorithm the maximum amount of times specified
        for w in range(self.__max_iterations):
            #Initiate cluster hypotheses to -1 since we don't know yet
            cluster_hypotheses = []

            #For each point let's get the distances to each centroid
            for x in X:
                curr_dist = sys.float_info.max
                prev_dist = sys.float_info.max
                d = sys.float_info.max
                
                for j, c in enumerate(centroids):
                
                    curr_dist = self.dist(x, c)
                    d = j if curr_dist < prev_dist else d
                    prev_dist = curr_dist
                cluster_hypotheses.append(d)
            
            potential_clusters = {}
            
            for hyp, x in zip(cluster_hypotheses, X):
                if hyp not in potential_clusters:
                    potential_clusters[hyp] = [x]
                else:
                    potential_clusters[hyp].append(x)
            
            for key in potential_clusters:
                centroids[key] = np.mean(potential_clusters[key], axis=0)

        
        #return(cluster_hypotheses)
        #print(cluster_hypotheses)

        return([cluster_hypotheses] + [centroids])