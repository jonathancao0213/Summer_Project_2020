import bs4
from urllib.request import urlopen
from bs4 import BeautifulSoup as soup
from lxml import etree
import csv
import sys
import requests

def get_news_for():
    # First from bloomberg.com
    url = "https://finance.yahoo.com/news/"
    print("Scraping from %s" % url)

    client = urlopen(url)
    html = client.read()
    client.close()

    page = soup(html, "html.parser")
    file = open("Data/news.txt", mode='a', newline='')

    #print(page.find_all("li", attrs={"class": "js-stream-content Pos(r)"}))
    news = page.find_all("li")
    for n in news:
        if n.has_attr("data-reactid"):
            if n.has_attr("aria-label"):
                dat = '|' + n.a["aria-label"]
            else:
                continue
        file.write(dat + '\n')

    news = page.find_all("p", {"class":"Fz(14px) Lh(19px) Fz(13px)--sm1024 Lh(17px)--sm1024 LineClamp(2,38px) LineClamp(2,34px)--sm1024 M(0)"})#[0].get_text())
    for n in news:
        line = n.get_text().replace('\n', '')
        file.write('|'+ line + '\n')
    file.close()



if __name__ == "__main__":
    get_news_for()
