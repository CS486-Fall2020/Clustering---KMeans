from cluster import cluster
import math
import random

class kmeans(cluster):

    #Constructor for kmeans class
    def __init__(self, k=5, max_iterations=100):
        self.__k = k
        self.__max_iterations = max_iterations

    #This is the euclidean distance between x and y
    def dist(self, x, y):
        return (math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)])))

        
    def fit(self, X):
        X = [[random.random() for x in range(3)] for y in range(100)]

        #Picking centroids randomly in the data
        centroids = random.choices(X, k=self.__k)

        print(centroids)