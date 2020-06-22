import sys, ast
from create_past_database import create_database
from add_day_to_database import add_day

"""
with open('stock_watch.csv', newline = '') as e:
    for line in e.readlines():
        array = line.split(',')
        ticker = array[1]
        tickerlist.append(ticker)

tickerlist = tickerlist[1:]
"""

def ui(apikey, ticker, s_avg, v_avg):
    create_database(apikey, ticker, s_avg, v_avg)

    add_day(apikey, ticker, stock_avg, vol_avg)

if __name__ == "__main__":
    ui(sys.argv[1], sys.argv[2])
