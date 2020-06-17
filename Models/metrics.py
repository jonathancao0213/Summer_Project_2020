import numpy as np
import pandas as pd
import math
from sklearn.metrics import adjusted_mutual_info_score

def load_data(data_path):
    """
    Associated test case: tests/test_data.py

    Reading and manipulating data is a vital skill for machine learning.

    This function loads the data in data_path csv into two numpy arrays:
    features (of size NxK) and targets (of size Nx1) where N is the number of rows
    and K is the number of features.

    data_path leads to a csv comma-delimited file with each row corresponding to a
    different example. Each row contains binary features for each example
    (e.g. chocolate, fruity, caramel, etc.) The last column indicates the label for the
    example how likely it is to win a head-to-head matchup with another candy
    bar.

    This function reads in the csv file, and reads each row into two numpy arrays.
    The first array contains the features for each row. For example, in candy-data.csv
    the features are:

    chocolate,fruity,caramel,peanutyalmondy,nougat,crispedricewafer,hard,bar,pluribus

    The second array contains the targets for each row. The targets are in the last
    column of the csv file (labeled 'class'). The first row of the csv file contains
    the labels for each column and shouldn't be read into an array.

    Example:
    chocolate,fruity,caramel,peanutyalmondy,nougat,crispedricewafer,hard,bar,pluribus,class
    1,0,1,0,0,1,0,1,0,1

    should be turned into:

    [1,0,1,0,0,1,0,1,0] (features) and [1] (targets).

    This should be done for each row in the csv file, making arrays of size NxK and Nx1.

    Args:
        data_path (str): path to csv file containing the data

    Output:
        features (np.array): numpy array of size NxK containing the K features
        targets (np.array): numpy array of size Nx1 containing the 1 feature.
        attribute_names (list): list of strings containing names of each attribute
            (headers of csv)
    """

    # Implement this function and remove the line that raises the error after.
    # f = open(data_path)
    # data = csv.reader(f)

    # Read column names from file
    cols = list(pd.read_csv(data_path, nrows =1))
    attribute_names = cols[1:-1]

    # Use list comprehension to remove the unwanted column in **usecol**
    features = np.asarray(pd.read_csv(data_path, usecols=[i for i in cols if i != 'Time' and i != 'Buy']))
    targets = np.asarray(pd.read_csv(data_path, usecols=['Buy']))

    features = features.astype(np.float)
    targets = targets.astype(np.float)

    return features, targets, attribute_names

def train_test_split(features, targets, fraction):
    """
    Split features and targets into training and testing, randomly. N points from the data
    sampled for training and (features.shape[0] - N) points for testing. Where N:

        N = int(features.shape[0] * fraction)

    Returns train_features (size NxK), train_targets (Nx1), test_features (size MxK
    where M is the remaining points in data), and test_targets (Mx1).

    Args:
        features (np.array): numpy array containing features for each example
        targets (np.array): numpy array containing labels corresponding to each example.
        fraction (float between 0.0 and 1.0): fraction of examples to be drawn for training

    Returns
        train_features: subset of features containing N examples to be used for training.
        train_targets: subset of targets corresponding to train_features containing targets.
        test_features: subset of features containing N examples to be used for testing.
        test_targets: subset of targets corresponding to test_features containing targets.
    """
    if (fraction > 1.0):
        raise ValueError('N cannot be bigger than number of examples!')

    if fraction == 1.0:
        train_features = features
        train_targets = targets
        test_features = train_features
        test_targets = train_targets
    else:
        f = math.floor(fraction*len(features))
        train_features = features[:f,:]
        train_targets = targets[:f]
        test_features = features[f:,:]
        test_targets = targets[f:]

    return train_features, train_targets, test_features, test_targets

def confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the confusion matrix. The confusion
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    YOU DO NOT NEED TO IMPLEMENT CONFUSION MATRICES THAT ARE FOR MORE THAN TWO
    CLASSES (binary).

    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    tn = 0
    fp = 0
    fn = 0
    tp = 0

    for index in range(0, len(actual)):
        if actual[index] == 1 and predictions[index] == 1:
            tp += 1
        elif actual[index] == 0 and predictions[index] == 0:
            tn += 1
        elif actual[index] == 1 and predictions[index] == 0:
            fn += 1
        else:
            fp += 1

    return [[tn, fp],[fn, tp]]

def accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the accuracy:

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    if len(actual) == 0:
        return 1

    [[tn,fp],[fn,tp]] = confusion_matrix(actual,predictions)
    return (tn + tp) / len(actual)

def precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    [[tn,fp],[fn,tp]] = confusion_matrix(actual,predictions)

    if tp == 0 and fp == 0:
        precision = 1
    else:
        precision = tp / (tp + fp)

    if tp == 0 and fn == 0:
        recall = 1
    else:
        recall = tp / (tp + fn)

    return precision, recall

def f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the F1-measure:

   https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Hint: implement and use the precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    precision, recall = precision_and_recall(actual,predictions)

    return 2 * precision * recall / (precision + recall)

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """

    M = len(X)
    if Y.ndim == 1:
        N = 1
        D = np.zeros((M,N))

        for m in range(0,M):
            D[m] = np.linalg.norm(X[m] - Y)
    else:
        N = len(Y)
        D = np.zeros((M,N))

        for m in range(0,M):
            for n in range(0,N):
                D[m,n] = np.linalg.norm(X[m] - Y[n])

    return D

def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """

    M = len(X)
    if Y.ndim == 1:
        N = 1
        D = np.zeros((M,N))

        for m in range(0,M):
            s = np.sum(np.absolute(X[m] - Y))
            D[m] = s
    else:
        N = len(Y)
        D = np.zeros((M,N))

        for m in range(0,M):
            for n in range(0,N):
                s = np.sum(np.absolute(X[m] - Y[n]))
                D[m,n] = s

    return D

def cosine_distances(X, Y):
    """Compute Cosine distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    M = len(X)
    if Y.ndim == 1:
        N = 1
        D = np.zeros((M,N))

        for m in range(0,M):
            dot = np.dot(X[m], Y)
            Ux = np.linalg.norm(X[m])
            Uy = np.linalg.norm(Y)

            s = dot / (Ux * Uy)

            D[m] = 1 - s
    else:
        N = len(Y)
        D = np.zeros((M,N))

        for m in range(0,M):
            for n in range(0,N):
                dot = np.dot(X[m], Y[n])
                Ux = np.linalg.norm(X[m])
                Uy = np.linalg.norm(Y[n])

                s = dot / (Ux * Uy)

                D[m,n] = 1 - s

    return D

def adjusted_mutual_info(predicted_labels, predicted_targets):
    """
    Adjusted mutual info score is a metric used to judge the quality of a clustering
    output when ground truth labels are available. We're just wrapping it here so you
    can use it in your experiments. Here's the documentation:

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html#sklearn.metrics.adjusted_mutual_info_score
    """
    return adjusted_mutual_info_score(predicted_labels, predicted_targets)
