import sys, ast, csv
from add_today import add_day

"""
with open('stock_watch.csv', newline = '') as e:
    for line in e.readlines():
        array = line.split(',')
        ticker = array[1]
        tickerlist.append(ticker)

tickerlist = tickerlist[1:]
"""

def ui(apikey):
    with open("Data/pair_ticker.csv", mode='r') as file:
        reader = csv.reader(file)
        data = list(reader)
    for company in data:
        add_day(apikey, company[0])

if __name__ == "__main__":
    ui(sys.argv[1])
