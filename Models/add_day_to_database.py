import sys
import numpy as np
import time
import datetime
import requests

def add_day(apikey, tickerlist):
    """
    This function adds onto an already existing database for a specific ticker.
    Adds today's data.

    Note that in order for this function to execute, you must have a TD Ameritrade
    account and provide an apikey to access the information

    apikey: apikey
    tickerlist: list of tickers in the format of ['TICKER1', 'TICKER2', etc]
    """
    for ticker in tickerlist:
        print("Adding today to the %s database" % ticker)

        link = 'https://api.tdameritrade.com/v1/marketdata/%s/quotes' % ticker
        ten_days_link = 'https://api.tdameritrade.com/v1/marketdata/%s/pricehistory' % ticker
        specs = {'apikey':apikey}
        ten_days_specs = {'apikey':apikey, 'period':10, 'periodType':'day', 'frequency':1, 'frequencyType':'daily'}
        today = requests.get(url = link, params = specs)
        ten_days = requests.get(url = ten_days_link, params = ten_days_specs)

        today_data = today.json()
        ten_days_data = ten_days.json()

        stock = today_data[ticker]
        print(stock)

        # ask = stock['askPrice']
        # bid = stock['bidPrice']
        # yearhigh = stock['52WkHigh']
        # yearlow = stock['52WkLow']
        # daylow = stock['lowPrice']
        # dayhigh = stock['highPrice']
        # dayopen = stock['openPrice']
        # dayclose = stock['closePrice']
        # v = stock['volatility']
        # time = datetime.date()

if __name__ == "__main__":
    add_day(sys.argv[1], sys.argv[2])
