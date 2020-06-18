import sys
import numpy as np
# All metrics defined in here
import metrics
# Decision Tree and Prior Probability (hw 1)
from decision_tree import DecisionTree
from prior_probability import PriorProbability
# K Nearest Neighbor (hw2)
from knn import KNearestNeighbor
# K Means and GMM clustering (hw 5)
from kmeans import KMeans
from gmm import GMM

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
    GOOGL_closing_data = features[:,5].reshape(-1,1)
    n = 3

    #Data Processing
    data0 = features[:,5]
    example0 = data0[:-n].reshape(-1,1)

    data1 = features[:,[5,6]]
    example1 = data1[:-n]

    target = GOOGL_closing_data[n:]

    #Train and Test
    train_features, train_targets, test_features, test_targets = metrics.train_test_split(example0, target, 0.8)
    train_features1, train_targets1, test_features1, test_targets1 = metrics.train_test_split(example1, target,0.8)
    # x_train0, x_test0, y_train0, y_test0 = train_test_split(example0,target, test_size = 0.2, random_state = 20)
    # x_train1, x_test1, y_train1, y_test1 = train_test_split(example1,target, test_size = 0.2, random_state = 20)
    lr = LinearRegression()
    lr.fit(train_features, train_targets)
    lr_confidence = lr.score(test_features, test_targets)
    print("R2 score:", lr_confidence)

    lr1 = LinearRegression()
    lr1.fit(train_features1, train_targets1)
    lr1_confidence = lr1.score(test_features1, test_targets1)
    print("R2 score1:", lr1_confidence)

elif model == "gradient_descent":
    raise NotImplementError()

elif model == "kmeans":
    # Need to make continuous for higher Mutual Info Score
    kmeans = KMeans(2)
    kmeans.fit(trainf)
    labels = kmeans.predict(testf)
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
    # Also need to make continuous
    gmm = GMM(2, 'spherical')
    gmm.fit(trainf)
    labels = gmm.predict(testf)
    #acc = metrics.adjusted_mutual_info(testt.flatten(), labels)
    print(np.subtract(labels,testt.flatten()))

    cm = metrics.confusion_matrix(testt.flatten(), labels)
    a = metrics.accuracy(testt.flatten(), labels)
    p, r = metrics.precision_and_recall(testt.flatten(), labels)
    f = metrics.f1_measure(testt.flatten(), labels)
    print("Accuracy = %f\n" % a)
    print("Precision = %f, Recall = %f\n" % (p,r))
    print("F1 measure = %f\n" % f)


elif model == "reinforcement":
    raise NotImplementError()

elif model == "neural_network":
    raise NotImplementError()

else:
    raise NotImplementError()
