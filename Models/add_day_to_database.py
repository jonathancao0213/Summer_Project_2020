import sys
import numpy as np
import time
import datetime
from datetime import datetime
import requests
import csv
from calculate_trend import first_derivative, second_derivative

def add_day(apikey, ticker):
    """
    This function adds onto an already existing database for a specific ticker.
    Adds today's data.

    Note that in order for this function to execute, you must have a TD Ameritrade
    account and provide an apikey to access the information

    apikey: apikey
    tickerlist: list of tickers in the format of ['TICKER1', 'TICKER2', etc]
    """
    #for i, ticker in enumerate(tickerlist):
    print("Adding today to the %s database" % ticker)

    with open("Data/%s_stock_normalized.csv" % ticker, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    stocks = []
    for i, point in enumerate(data):
        if i == 0:
            continue
        stocks.append(float(point[1]))
    stocks = stocks[1:]

    stock_moving_average = sum(stocks)/len(stocks)

    if data[-1][0] == datetime.strftime(datetime.now(), "%Y-%m-%d"):
        print("Database was just updated with today's %s data, moving on to next ticker" % ticker)
        return 0

    link = 'https://api.tdameritrade.com/v1/marketdata/%s/quotes' % ticker
    # history_link = 'https://api.tdameritrade.com/v1/marketdata/%s/pricehistory' % ticker
    specs = {'apikey':apikey}
    # month_specs = {'apikey':apikey, 'period':1, 'periodType':'month', 'frequency':1, 'frequencyType':'daily'}
    today = requests.get(url = link, params = specs)
    month = requests.get(url = history_link, params = month_specs)

    today_data = today.json()
    month_data = month.json()

    stock = today_data[ticker]
    two_weeks_data = data[-10:]

    yesterday_close_to_today_open, past_first_derivative, past_avg_normalized_open_to_close = first_derivative(two_weeks_data)
    past_curve = second_derivative(two_weeks_data)

    yearhigh = stock['52WkHigh']
    yearlow = stock['52WkLow']
    dayopen = stock['openPrice']
    dayclose = stock['closePrice']
    volume = stock['totalVolume']

    if dayopen > close:
        from_close_to_open = 1
    else:
        from_close_to_open = 0

    if dayopen > stock_avg[i]:
        open_to_moving_average = 1
    else:
        open_to_moving_average = 0

    if dayopen > (yearhigh + yearlow)/2:
        open_to_year_average = 1
    else:
        open_to_year_average = 0

    if volume > vol_avg[i]:
        volume_to_moving_average = 1
    else:
        volume_to_moving_average = 0

    if dayclose > dayopen:
        buy = 1
    else:
        buy = 0

    file = open('Data/%s_stock_discrete.csv' % ticker, mode='a', newline='')
    writer = csv.writer(file, delimiter=',')
    writer.writerow([datetime.strftime(datetime.now(), "%Y-%m-%d"), \
    prev_result, past_day_trend, past_trend, past_curve, from_close_to_open, \
    open_to_moving_average, open_to_year_average, volume_to_moving_average, buy])

    updated_stock_avg = (stock_avg[i]*len(data) + dayopen)/(len(data) + 1)
    updated_vol_avg = (vol_avg[i]*len(data) + volume)/(len(data) + 1)

    return updated_stock_avg, updated_vol_avg

if __name__ == "__main__":
    add_day(sys.argv[1], sys.argv[2])
