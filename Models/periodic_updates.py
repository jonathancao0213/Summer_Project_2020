import sys, ast, csv
from add_today import add_day
from timeloop import Timeloop
import time
import numpy as np
from datetime import timedelta
from datetime import datetime
from newscrawl import get_news_for, check_if_news_exists


"""
with open('stock_watch.csv', newline = '') as e:
    for line in e.readlines():
        array = line.split(',')
        ticker = array[1]
        tickerlist.append(ticker)

tickerlist = tickerlist[1:]
"""

apikey = None
tl = Timeloop()

@tl.job(interval=timedelta(hours=5))
def update_stock():
    if datetime.today().weekday() == 5 or datetime.today().weekday() == 6:
        print("Today is the weekend, and there will be no updated stock")
    else:
        with open("Data/pair_ticker.csv", mode='r') as file:
            reader = csv.reader(file)
            data = list(reader)
        for company in data:
            add_day(apikey, company[0])
        print("Finished adding today's data to databases\n")

news_time = 6 + np.random.randn()
@tl.job(interval=timedelta(hours=news_time))
def update_news():
    print("Scraping the news in the past %d hours" % news_time)
    get_news_for()
    print("Finished adding last %d hours news to database\n" % news_time)

if __name__ == "__main__":
    apikey = sys.argv[1]
    tl.start(block=True)
