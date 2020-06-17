import sys
from numpy import genfromtxt
import numpy as np
# All metrics defined in here
import metrics
# Decision Tree and Prior Probability (hw 1)
from decision_tree import DecisionTree
from prior_probability import PriorProbability
# K Nearest Neighbor (hw2)
from knn import KNearestNeighbor
# K Means clustering (hw 5)
from kmeans import KMeans

data = sys.argv[1]

features, targets, attribute_names = metrics.load_data("Data/" + data)

model = sys.argv[2]

fraction = (float)(sys.argv[3])

trainf, traint, testf, testt = metrics.train_test_split(features, targets, fraction)

if model == "decision_tree":
    tree = DecisionTree(attribute_names)
    print(attribute_names)
    tree.fit(trainf, traint)
    num_nodes, max_depth = tree.tree_attributes(tree.tree)
    t = tree.predict(testf)
    cm = metrics.confusion_matrix(testt, t)
    a = metrics.accuracy(testt, t)
    p, r = metrics.precision_and_recall(testt, t)
    f = metrics.f1_measure(testt, t)
    print("Accuracy = %f\n" % a)
    print("Precision = %f, Recall = %f\n" % (p,r))
    print("F1 measure = %f\n" % f)

elif model == "prior_probability":
    prob = PriorProbability()
    prob.fit(trainf, traint)
    t = prob.predict(testf)
    #raise ValueError(t)
    cm = metrics.confusion_matrix(testt, t)
    a = metrics.accuracy(testt, t)
    p, r = metrics.precision_and_recall(testt, t)
    f = metrics.f1_measure(testt, t)
    print("Accuracy = %f\n" % a)
    print("Precision = %f, Recall = %f\n" % (p,r))
    print("F1 measure = %f\n" % f)

elif model == "knn":
    knn = KNearestNeighbor(10, distance_measure='euclidean', aggregator='mean')
    knn.fit(trainf, traint)

    labels = knn.predict(testf)
    binary_labels = []
    for each in labels:
        if each > 0.5:
            binary_labels.append(1)
        else:
            binary_labels.append(0)

    binary_labels = np.asarray(binary_labels)

    cm = metrics.confusion_matrix(testt, binary_labels)
    a = metrics.accuracy(testt, binary_labels)
    p, r = metrics.precision_and_recall(testt, binary_labels)
    f = metrics.f1_measure(testt, binary_labels)
    print("Accuracy = %f\n" % a)
    print("Precision = %f, Recall = %f\n" % (p,r))
    print("F1 measure = %f\n" % f)

elif model == "linear_regression":
    raise NotImplementError()

elif model == "gradient_descent":
    raise NotImplementError()

elif model == "kmeans":
    m = KMeans(2)
    m.fit(trainf)
    labels = m.predict(testf)
    #acc = metrics.adjusted_mutual_info(testt.flatten(), labels)
    print(np.subtract(labels,testt.flatten()))

    cm = metrics.confusion_matrix(testt.flatten(), labels)
    a = metrics.accuracy(testt.flatten(), labels)
    p, r = metrics.precision_and_recall(testt.flatten(), labels)
    f = metrics.f1_measure(testt.flatten(), labels)
    print("Accuracy = %f\n" % a)
    print("Precision = %f, Recall = %f\n" % (p,r))
    print("F1 measure = %f\n" % f)

elif model == "gmm":
    raise NotImplementError()

elif model == "reinforcement":
    raise NotImplementError()

elif model == "neural_network":
    raise NotImplementError()

else:
    raise NotImplementError()
