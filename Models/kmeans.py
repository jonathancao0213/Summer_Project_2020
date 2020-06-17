import numpy as np

class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        max_iter = 2000

        targets = np.zeros(features.shape[0])

        center = features.mean(0)

        means = np.random.uniform(center - 0.5, center + 0.5, (self.n_clusters, features.shape[1]))
        #means = np.random.rand(self.n_clusters, features.shape[1])
        prev = np.zeros((self.n_clusters, features.shape[1]))

        iter = 0
        while iter <= max_iter:
            if abs(np.sum(means - prev)) <= 1e-3:
                break

            prev = means
            for i, value in enumerate(features): # loop through each point in feature to compare against each mean
                d = np.zeros(means.shape[0]) # list of zeros for the distances
                for j, comp in enumerate(means): # looping through the means
                    distance = np.linalg.norm(value - comp) # calculate distance between the point and the mean
                    d[j] = distance # set d[j] to be the euclidean distance
                closest = np.where(d == np.amin(d))[0][0] # find the index at which the distance is shortest (essentially the label of the point)
                targets[i] = closest # set the targets matrix to be the index at which the distance is shortest

            for n in range(0, self.n_clusters): # calculate new mean and update
                a = np.where(targets == n) # find index of every value with targets = n
                wanted_features = features[a]
                s = np.zeros(features.shape[1])
                for f in wanted_features: # for each of their features, find average
                    s += f

                if len(wanted_features) == 0:
                    means[n] = means[n]
                else:
                    avg = s/len(wanted_features)
                    means[n] = avg

            iter += 1

        self.means = means

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """

        predictions = np.zeros(features.shape[0])

        for i, value in enumerate(features):
            min = np.inf

            for j in range(0,self.n_clusters):
                distance = np.linalg.norm(self.means[j] - value)
                if distance < min:
                    min = distance
                    predictions[i] = j

        return predictions
