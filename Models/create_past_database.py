import requests
import sys
import os.path
from os import path
import csv
import signal
import datetime
import pandas as pd
from calculate_trend import first_derivative, second_derivative

# Set up keyboard interrupt to stop updating the csv file
# No longer needed but just kept for future reference
"""
def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Stopping stock watch...".format(signal))
    exit(0)

signal.signal(signal.SIGINT, keyboardInterruptHandler)
"""

def create_database(apikey, tickerlist, replace=False):
    """
    This function creates a database that contains the past year's stock data for companies in tickerlist.

    Note that in order for this function to execute, you must have a TD Ameritrade
    account and provide an apikey to access the information

    apikey: apikey
    tickerlist: list of tickers in the format of ['TICKER1', 'TICKER2', etc]
    replace: if replace is True, then even if the database with the ticker already exists,
        the program will still execute and replace it
    """

    for j, ticker in enumerate(tickerlist):
        print("Creating %s database..." % ticker)
        """
        Checking if the database already exists and isn't empty.
        If the database exists and isn't empty and replace == False, then we
            move on to the next ticker
        If the database exists and replace == True,
            then we replace it by executing the program.
        Otherwise we create the database.
        """
        if path.exists('Data/%s_stock.csv' % ticker) and replace == False:
            df = pd.read_csv('Data/%s_stock.csv' % ticker)
            if not df.empty:
                print("Database containing %s already exists, moving to next ticker." % ticker)
                continue

        # Get price history
        link = 'https://api.tdameritrade.com/v1/marketdata/%s/quotes' % ticker
        historylink = 'https://api.tdameritrade.com/v1/marketdata/%s/pricehistory' % ticker

        specs = {'apikey':apikey}
        history_specs = {'apikey':apikey, 'period':1, 'periodType':'year', 'frequency':1, 'frequencyType':'daily'}

        overall = requests.get(url = link, params = specs)
        history = requests.get(url = historylink, params = history_specs)

        overall_data = overall.json()
        history_data = history.json()

        yearhigh = overall_data[ticker]['52WkHigh']
        yearlow = overall_data[ticker]['52WkLow']

        stock_moving_average = 0
        volume_moving_average = 0
        prev_result = 0
        prev_close = 0

        file = open('Data/%s_stock.csv' % ticker, mode='w')
        writer = csv.writer(file, delimiter=',')

        writer.writerow(['Time', 'Previous Trend', 'Yesterday Close to Today Open', 'Open to Moving Average', 'Open to 52 Wk Average', 'Volume to Moving Average', 'Buy'])

        for i, row in enumerate(history_data['candles']):
            dayopen = row['open']
            dayhigh = row['high']
            daylow = row['low']
            dayclose = row['close']
            volume = row['volume']
            t = row['datetime']/1000.0
            time = datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S.%f')

            if prev_close <= dayopen: # checking if the price went up from yesterday's close to today's open
                from_close_to_open = 1
            else:
                from_close_to_open = 0

            if dayopen > stock_moving_average: # check if the price is higher than moving average
                open_to_moving_average = 1
            else:
                open_to_moving_average = 0

            if dayopen > (yearhigh + yearlow) / 2: # check if the price is higher than 52wk average
                open_to_year_average = 1
            else:
                open_to_year_average = 0

            if volume > volume_moving_average: # check if the volume is higher than the moving average
                volume_to_moving_average = 1
            else:
                volume_to_moving_average = 0

            if dayclose > dayopen: # checking if day close is greater than day open
                buy = 1
            else:
                buy = 0
            writer.writerow([time, prev_result, from_close_to_open, open_to_moving_average, open_to_year_average, volume_to_moving_average, buy])

            # previous day's trend is whether it went up or down
            prev_result = buy

            # update previous close
            prev_close = dayclose

            #update moving volume average
            totalv = volume_moving_average * i
            totalv += volume
            volume_moving_average = totalv / (i+1)

            # update moving dayopen average
            total = stock_moving_average * i
            total += dayopen
            stock_moving_average = total / (i+1)

            # with open('%s_stock.csv' % ticker, mode='w') as file:
            #     writer = csv.writer(file, delimiter=',')
            #     writer.writerow([time, yearhigh, yearlow, dayhigh, daylow, dayopen, dayclose, volume, v])

        print("Database for %s has been created." % ticker)

    print("Completed creating databases.\n")

if __name__ == "__main__":
    create_database(sys.argv[1], sys.argv[2], replace=True)
