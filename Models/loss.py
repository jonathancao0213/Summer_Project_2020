import numpy as np

"This is only for linear models thus far"

class Loss:
    def __init__(self,regularization=None):
        self.regularization = regularization

    def loss(self, X, y, w):
        pass
    def gradient(self,X, y, w):
        pass

class SquaredLoss(Loss):

    "Squaure Loss Formula = (1/2N) summation_i_to_N((y - w^T x)^2)"

    def loss(self, X, y, w):
        wTX = np.dot(w,np.transpose(X))
        squared = 0.5*np.square(y-wTX)
        # print(len(squared))
        return np.sum(squared)/len(squared)

    def gradient(self,X, y, w):
        wTX0 = np.dot(w,np.transpose(X))
        wTX = wTX0.reshape(-1,1)
        dot = np.dot(np.transpose(y-wTX), X)
        # print(len(X))
        return -dot/len(X)
