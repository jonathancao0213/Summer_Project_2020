import numpy as np
from numpy import genfromtxt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Convert Data from CSV file to numpy
df =  genfromtxt("Data/GOOGL_stock.csv", delimiter = ",")

#Get Closing Price Data
GOOGL_closing_data = df[1:,6]

# Number of days into future to predict
n = 10

#example up to the n-th days
example = GOOGL_closing_data[:-n].reshape(-1,1)

#targets is the closing price n days after current date
target = GOOGL_closing_data[n:]

#Split training and testing data
x_train, x_test, y_train, y_test = train_test_split(example,target, test_size = 0.2)

#Create and Train Linear Regression Model
lr = LinearRegression()
lr.fit(x_train, y_train)

#Test Model
lr_confidence = lr.score(x_test,y_test)
print("Confidence:",lr_confidence)

#Predict the Next n days
last_ndays = GOOGL_closing_data[-n:].reshape(-1,1)
# print(last_ndays)
lr_prediction = lr.predict(last_ndays)
print(lr_prediction)
