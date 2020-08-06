import sys, ast, csv
from add_today import add_day
from create_database import create_database

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
        #create_database(apikey, company[0], True)
        add_day(apikey, company[0])
    print("Finished adding today's data to databases")

if __name__ == "__main__":
    ui(sys.argv[1])
