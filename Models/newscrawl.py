import bs4
from urllib.request import urlopen
from bs4 import BeautifulSoup as soup
import csv
import sys
import requests

def get_news_for(ticker):
    # First from bloomberg.com
    url = "https://www.bloomberg.com/"

    client = urlopen(url)
    html = client.read()
    client.close()

    page = soup(html, "html.parser")

    #print(page)

    #print(page.head) # prints the <head></head>

def pair_ticker(ticker, name):
    with open("Data/pair_ticker.csv", mode='r') as read_file:
        reader = csv.reader(read_file)
        f = list(reader)

    for row in f:
        if row.count(ticker) > 0:
            print("Ticker (%s) and company name (%s) already exists" % (ticker, row[1]))
            return

    file = open("Data/pair_ticker.csv", mode='a', newline='')
    writer = csv.writer(file, delimiter=',')
    name = name.split(" - ")
    writer.writerow([ticker, name[0]])



if __name__ == "__main__":
    apikey = sys.argv[1]
    ticker = sys.argv[2]
    link = 'https://api.tdameritrade.com/v1/marketdata/%s/quotes' % ticker
    specs = {'apikey':apikey}
    d = requests.get(url = link, params = specs)
    data = d.json()
    pair_ticker(ticker, data[ticker]['description'])
