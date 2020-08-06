import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from math import ceil

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
# Regression
from linear_regression import LinearRegression
# GD
from gradient_descent import GradientDescent
# Neural network
from neural_network import Stock_Classifier, run_model

data = sys.argv[1]

features, targets, attribute_names = metrics.load_data("Data/" + data)
today = features[-3:-1,:]

model = sys.argv[2]

fraction = (float)(sys.argv[3])

trainf, traint, testf, testt = metrics.train_test_split(features, targets, fraction)
btrainf = np.ceil(np.asarray(trainf)/abs(np.asarray(trainf)))
btestf = np.ceil(np.asarray(testf)/abs(np.asarray(testf)))

# btrainf = btrainf.tolist()
# btestf = btestf.tolist()


if model == "decision_tree":
    tree = DecisionTree(attribute_names)
    tree.fit(btrainf, traint)
    num_nodes, max_depth = tree.tree_attributes(tree.tree)
    t = tree.predict(btestf)
    decision = tree.predict(today)
    #print(decision)
    #print("Should you buy tomorrow with today's data? %d" % decision)
    cm = metrics.confusion_matrix(testt, t)
    a = metrics.accuracy(testt, t)
    p, r = metrics.precision_and_recall(testt, t)
    f = metrics.f1_measure(testt, t)
    print("Accuracy = %f\n" % a)
    print("Precision = %f, Recall = %f\n" % (p,r))
    print("F1 measure = %f\n" % f)
    print("Tomorrow's Forecast: %f\n" % decision[-1])

elif model == "prior_probability":
    prob = PriorProbability()
    print(btrainf)
    prob.fit(btrainf, traint)
    t = prob.predict(btestf)
    decision = prob.predict(today)
    #raise ValueError(t)
    cm = metrics.confusion_matrix(testt, t)
    a = metrics.accuracy(testt, t)
    p, r = metrics.precision_and_recall(testt, t)
    try:
        f = metrics.f1_measure(testt, t)
    except:
        f = 0
    print("Accuracy = %f\n" % a)
    print("Precision = %f, Recall = %f\n" % (p,r))
    print("F1 measure = %f\n" % f)
    print("Tomorrow's Forecast: %f\n" % decision[-1])
    print(sum(t)/len(t))


elif model == "knn":
    knn = KNearestNeighbor(10, distance_measure='euclidean', aggregator='mean')
    knn.fit(trainf, traint)

    labels = knn.predict(testf)
    binary_labels = metrics.make_binary(labels)

    cm = metrics.confusion_matrix(testt, binary_labels)
    a = metrics.accuracy(testt, binary_labels)
    p, r = metrics.precision_and_recall(testt, binary_labels)
    f = metrics.f1_measure(testt, binary_labels)
    print(binary_labels)
    print("Accuracy = %f\n" % a)
    print("Precision = %f, Recall = %f\n" % (p,r))
    print("F1 measure = %f\n" % f)
    print(sum(binary_labels)/len(binary_labels))

elif model == "linear_regression":

    lr = LinearRegression()
    lr.fit(trainf, traint)
    lrc = lr.score(testf, testt)
    labels = lr.predict(testf)
    binary_labels = metrics.make_binary(labels)
    print(binary_labels)
    a = metrics.accuracy(testt, binary_labels)
    print("Accuracy = %f\n" % a)
    print("R2 sore:", lrc)


    # GOOGL_closing_data = features[:,5].reshape(-1,1)
    # n = 3
    #
    # #Data Processing
    # data0 = features[:,5]
    # example0 = data0[:-n].reshape(-1,1)
    #
    # data1 = features[:,[5,6]]
    # example1 = data1[:-n]
    #
    # target = GOOGL_closing_data[n:]
    #
    # #Train and Test
    # train_features, train_targets, test_features, test_targets = metrics.train_test_split(example0, target, 0.8)
    # train_features1, train_targets1, test_features1, test_targets1 = metrics.train_test_split(example1, target,0.8)
    # lr = LinearRegression()
    # lr.fit(train_features, train_targets)
    # lr_confidence = lr.score(test_features, test_targets)
    # print("R2 score:", lr_confidence)
    #
    # lr1 = LinearRegression()
    # lr1.fit(train_features1, train_targets1)
    # lr1_confidence = lr1.score(test_features1, test_targets1)
    # print("R2 score1:", lr1_confidence)

elif model == "gradient_descent":
    GOOGL_closing_data = features[:,5].reshape(-1,1)
    n = 3

    #Data Processing
    data0 = features[:,5]
    example0 = data0[:-n].reshape(-1,1)
    target = GOOGL_closing_data[n:]


    #Train and Test
    train_features, train_targets, test_features, test_targets = metrics.train_test_split(example0, target, 0.8)
    gd = GradientDescent()
    gd.fit(train_features, train_targets)
    gd_confidence = gd.score(test_features, test_targets)
    print("R2 score:", gd_confidence)

elif model == "kmeans":
    # Need to make continuous for higher Mutual Info Score
    kmeans = KMeans(2)
    kmeans.fit(trainf)
    labels = kmeans.predict(testf)
    #acc = metrics.adjusted_mutual_info(testt.flatten(), labels)
    print(labels)

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
    print(labels)

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
    train_dataset = metrics.MyDataset(trainf, traint)
    valid_dataset = metrics.MyDataset(testf, testt)

    running_mode = 'test'

    model = Stock_Classifier(len(trainf[0])) # Change this line to Stock_Classifier
    _m, _est_loss, _est_acc, predictions = run_model(model,running_mode=running_mode, train_set=train_dataset,valid_set = valid_dataset, batch_size=1, learning_rate=1e-3, n_epochs=5, shuffle=False)

    # train_loader = DataLoader(train_dataset)
    # for i, (inputs, targets) in enumerate(train_loader):
    #     outputs = Basic_Model(inputs)
    #     targets = targets.long()
    #     print([outputs, targets])

    if running_mode == 'train':
        _est_loss_train = np.mean(_est_loss['train'])
        _est_loss_valid = np.mean(_est_loss['valid'])

        _est_acc_train = np.mean(_est_acc['train'])
        _est_acc_valid = np.mean(_est_acc['valid'])

        _est_values = np.array([_est_loss_train, _est_loss_valid, _est_acc_train, _est_acc_valid])

        print(_est_values)
    else:
        print(predictions)
        print([_est_loss, _est_acc])

else:
    print("Enter valid model")
