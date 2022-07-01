"""
Implement the KMeans model 
data: is a numpy array
"""
import numpy as np

class KMeans:
    #  initalize the key parameters 
    def __init__(self, k = 2, tol = 0.0001, max_iter = 300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    
    # fit method 
    def fit(self, data):
        self.centroids = {}

        # initalized centroids 
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for feature in data:
                distance = np.sum([(feature - self.centroids[centroid])**2 for centroid in self.centroids], axis = 1)
                cluster = np.argmin(distance)
                self.classifications[cluster].append(feature)

            prev_centroids = dict(self.centroids)

            for cluster in self.classifications:
                self.centroids[cluster] = np.mean(self.classifications[cluster], axis = 0)

            optimized = True

            for c in self.centroids:
                prev = prev_centroids[c]
                current = self.centroids[c]

                if np.sum(abs(current - prev)) > self.tol:
                    optimized = False
                    break

            if optimized:
                break

    # predict method 
    def predict(self, data):
        distance = np.sum([(data - self.centroids[centroid])**2 for centroid in self.centroids], axis = 1)
        prediction = np.argmin(distance)

        return prediction 


# Example 
X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11]]
            )

y = np.array([7,7])

model_kmeans = KMeans()
model_kmeans.fit(X)
print(model_kmeans.centroids)
print(model_kmeans.predict(y))
