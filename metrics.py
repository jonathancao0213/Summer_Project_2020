import numpy as np

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
