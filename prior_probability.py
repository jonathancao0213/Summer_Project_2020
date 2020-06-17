import numpy as np
import statistics

class PriorProbability():
    def __init__(self):
        """
        This is a simple classifier that only uses prior probability to classify
        points. It just looks at the classes for each data point and always predicts
        the most common class.

        """
        self.most_common_class = None

    def fit(self, features, targets):
        """
        Implement a classifier that works by prior probability. Takes in features
        and targets and fits the features to the targets using prior probability.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        """
        """
        prob = np.empty(len(features[0]))

        for index in range(0, len(prob)):
            feat_ones = 0
            one_ones = 0
            for row in range(0,len(features)):
                if features[row,index] == 1:
                    feat_ones += 1
                    if targets[row] == 1:
                        one_ones += 1

            prob.append(one_ones/feat_ones)
        """
        s = sum(targets)
        if s >= len(targets)/2:
            m = 1
        else:
            m = 0

        self.most_common_class = m

    def predict(self, data):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        """
        """
        targets = np.empty(1)

        for row in data:
            total = sum(row)
            total_prob = np.dot(row, prob)
            if total_prob/total > 0.5:
                result = 1
            else:
                result = 0
            targets.append(result)
        """
        N = len(data)
        if self.most_common_class == 1:
            targets = np.ones(N)
        else:
            targets = np.zeros(N)

        return targets
