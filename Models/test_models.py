import sys
from numpy import genfromtxt
import numpy as np
# All metrics defined in here
import metrics
# Decision Tree and Prior Probability (hw 1)
from decision_tree import DecisionTree
from prior_probability import PriorProbability
# from k_nearest_neighbor import KNearestNeighbor

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
    num_nodes = 0
    max_depth = 0
    print("Accuracy = %f\n" % a)
    print("Precision = %f, Recall = %f\n" % (p,r))
    print("F1 measure = %f\n" % f)

elif model == "knn":
    raise NotImplementError()

elif model == "linear_regression":
    raise NotImplementError()

elif model == "gradient_descent":
    raise NotImplementError()

elif model == "clustering":
    raise NotImplementError()

elif model == "reinforcement":
    raise NotImplementError()

elif model == "neural_network":
    raise NotImplementError()

else:
    raise NotImplementError()
