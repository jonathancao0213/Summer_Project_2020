import numpy
import math

"""
Functions used to compute past trends (i.e. first and second derivative)
Data is a numpy array containing stock data from td ameritrade api
# of Rows: Number of data points
Columns need to contain and be in the order of:
open, high, low, close, volume, datetime
"""

def first_derivative(data):
    open_to_close = []
    all = []
    derivatives = []
    for val in data:
        all.append(val['open'])
        all.append(val['close'])

    for i, val in enumerate(all):
        if i%2 == 0:
            open_to_close.append(all[i+1] - val)

        if i == 0 or i == len(all) - 1:
            pass
        else:
            d = (all[i+1] - all[i-1]) / 2
            d = float(str(round(d, 5)))
            derivatives.append(d)

    avg_prime = sum(derivatives)/len(derivatives)
    avg_day = sum(open_to_close)/len(open_to_close)
    return all[-1], open_to_close[-1], avg_prime, avg_day


def second_derivative(data):
    all = []
    derivatives2 = []
    for val in data:
        all.append(val['open'])
        all.append(val['close'])

    for i, val in enumerate(all):
        if i == 0 or i == len(all) - 1:
            pass
        else:
            d = all[i+1] + all[i-1] - 2*val
            d = float(str(round(d, 5)))
            derivatives2.append(d)

    avg_prime2 = sum(derivatives2)/len(derivatives2)
    return avg_prime2
