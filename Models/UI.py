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

def ui(apikey, tickerlist):
    create_database(apikey, tickerlist)

    add_day(apikey, tickerlist)

if __name__ == "__main__":
    tickerlist = ast.literal_eval(sys.argv[2])
    print(tickerlist)
    ui(sys.argv[1], tickerlist)
