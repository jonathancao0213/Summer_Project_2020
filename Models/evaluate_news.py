import numpy as np
from bayes_classifier import Bayes_Classifier, f_score
import sys

data = []

def load_data(d):
    global data
    f = open('Data/%s' % d, "r", encoding='utf8', errors='ignore')
    data = f.readlines()
    f.close()

def bayes():
    classifier = Bayes_Classifier()
    classifier.train(data[:599])
    actual, predictions = classifier.classify(data[:599])
    fpos, fneg, fneu = f_score(actual, predictions)
    print(fpos, fneg, fneu)



if __name__ == "__main__":
    load_data(sys.argv[1])
    bayes()
