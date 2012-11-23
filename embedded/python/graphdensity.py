from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import manhattan_distances
import numpy as np

class Density:

    def __init__(self, X, n_neighbors = 10, theta = 1):
        self.neigh = NearestNeighbors(p=1)
        self.neigh.fit(X)
        self.Pij = - self.neigh.kneighbors_graph(X, n_neighbors, mode='distance') / (2 *
                                                                     theta**2)
        self.Pij.data[:] = np.exp(self.Pij.data)
        self.Wij = self.Pij.sum(0)
        counts = np.bincount(self.Pij.indices, minlength = self.Pij.shape[0])
        counts[np.where(counts ==0)[0]] = 1
        #print(counts)
        self.Gra = np.array(self.Wij / counts).reshape(-1)

    def pick(self,i):
        indices = self.Pij.getrow(i).indices
        temp = self.Gra[i]
        self.Gra[indices] = self.Gra[indices] - temp
        return temp
    
    def getDensity(self):
        return self.Gra
