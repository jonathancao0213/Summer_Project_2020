import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split

class LinearRegression():
    def __init__(self):
        self.weights = None

    #Uses close-form solution: w = ((X^T*X)^(-1))*X^T*y
    def fit(self, features, targets):
        ones = np.ones((len(features),1))
        features_new = np.concatenate((ones,features),axis = 1)
        transpose_x = np.transpose(features_new)
        XT_X_inv = np.linalg.inv(np.matmul(transpose_x,features_new))
        XT_X_Inv_XT = np.matmul(XT_X_inv,transpose_x)
        self.weights = np.matmul(XT_X_Inv_XT, targets)

    #Apply weights to the independent variables
    def predict(self,features):
        ones = np.ones((len(features),1))
        features_new = np.concatenate((ones,features),axis = 1)
        return np.matmul(features_new,self.weights)

    #Return the coefficient of determination R^2 of the prediction
    def score(self, features, targets):
        prediction = self.predict(features)
        u = np.sum((targets-prediction)**2)
        v = np.sum((targets-np.mean(targets))**2)
        return (1-(u/v))

#Convert Data from CSV file to numpy
df =  genfromtxt("Data/GOOGL_stock.csv", delimiter = ",")

#Get Closing Price Data
GOOGL_closing_data = df[1:,6]

# Number of days into future to predict
n = 3

#example up to the n-th days
example = GOOGL_closing_data[:-n].reshape(-1,1)

#targets is the closing price n days after current date
target = GOOGL_closing_data[n:]

#Split training and testing data
x_train, x_test, y_train, y_test = train_test_split(example,target, test_size = 0.2)

#Create and Train Linear Regression Model
lrz = LinearRegression()
lrz.fit(x_train, y_train)


#Test Model
lr_confidence = lrz.score(x_test, y_test)
print("Confidence:", lr_confidence)

#Predict the Next n days
last_ndays = GOOGL_closing_data[-n:].reshape(-1,1)
# print(last_ndays)
lr_prediction = lrz.predict(last_ndays)
print(lr_prediction)
