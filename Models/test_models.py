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
df = genfromtxt("Data/" + data, delimiter=',')
data = df[1:, 1:]
features = data[:,:-1]
targets = data[:, -1]

model = sys.argv[2]

if model == "decision_tree":
    tree = DecisionTree(attribute_names)
    tree.fit(trainf, traint)
    num_nodes, max_depth = tree.tree_attributes(tree.tree)
    t = tree.predict(testf)
    cm = metrics.confusion_matrix(testt, t)
    a = metrics.accuracy(testt, t)
    p, r = metrics.precision_and_recall(testt, t)
    f = metrics.f1_measure(testt, t)

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
