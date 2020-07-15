import numpy as np
from bayes_classifier import Bayes_Classifier, f_score
from newscrawl import get_news_for
import sys

data = []
classifier = Bayes_Classifier()

def load_data(d):
    global data
    f = open('Data/%s' % d, "r", encoding='utf8', errors='ignore')
    data = f.readlines()
    f.close()

def bayes(d, amount=None):
    global classifier
    classifier.train(d[:599])
    actual, predictions = classifier.classify(d[:599])
    fpos, fneg, fneu = f_score(actual, predictions)
    return fpos, fneg, fneu

def eval_news_for(source):
    d = get_news_for(source)
    global classifier
    try:
        a, predictions = classifier.classify(d)
        print(predictions)
        l = list(zip(d,predictions))
        for each in l:
            print(each)
    except:
        print("Something went wrong in classify")

if __name__ == "__main__":
    # set = sys.argv[1]
    # amount of data used = sys.argv[2]
    # company name = sys.argv[3]

    load_data(sys.argv[1])
    try:
        amount = sys.argv[2]
        fpos, fneg, fneu = bayes(data, amount)
    except:
        fpos, fneg, fneu = bayes(data)

    print("Positive Fscore = %f, Negative Fscore = %f, Neutral Fscore = %f" % (fpos, fneg, fneu))

    try:
        company = sys.argv[3]
        eval_news_for(company)
    except:
        print("something went wrong in eval_news_for")
        pass
