from numpy import genfromtxt

"""Convert Data from CSV file to numpy"""

def GenerateData():
    df = genfromtxt(Data/GOOGL_stock.csv, delimiter = ",")
    GOOGL_data = df[1:,1:]

    # asdas

    features = data[:,:-1]
    target = data[:,-1]

    return features,target
