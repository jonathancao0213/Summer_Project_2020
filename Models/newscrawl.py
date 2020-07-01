import bs4
from urllib.request import urlopen
from bs4 import BeautifulSoup as soup
from lxml import etree
import csv
import sys
import requests
import signal

# def keyboardInterruptHandler(signal, frame):
#     print("KeyboardInterrupt (ID: {}) has been caught. Stopping news watch...".format(signal))
#     exit(0)
#
# signal.signal(signal.SIGINT, keyboardInterruptHandler)

def get_news_for(source=None):
    # First from bloomberg.com

    url = "https://finance.yahoo.com/news/"
    print("Scraping: " + url)

    client = urlopen(url)
    html = client.read()
    client.close()

    page = soup(html, "html.parser")

    read_file = open("Data/news.txt", mode='r')
    data = list(read_file)
    file = open("Data/news.txt", mode='a', newline='')

    #print(page.find_all("li", attrs={"class": "js-stream-content Pos(r)"}))

    # news = page.find_all("li")
    # for n in news:
    #     if n.has_attr("data-reactid"):
    #         if n.has_attr("aria-label"):
    #             dat = '|' + n.a["aria-label"]
    #         else:
    #             continue
    #
    #     first_sentence = dat.split('.')[0]
    #     if check_if_news_exists(first_sentence, data) == False:
    #         file.write(dat + '\n')

    news = page.find_all("p", {"class":"Fz(14px) Lh(19px) Fz(13px)--sm1024 Lh(17px)--sm1024 LineClamp(2,38px) LineClamp(2,34px)--sm1024 M(0)"})#[0].get_text())
    for n in news:
        line = n.get_text().replace('\n', '')
        first_sentence = line.split('.')[0]
        if check_if_news_exists(first_sentence, data) == False and "Final Round" not in line and "Akiko Fujita" not in line:
            print(line)
            bad = ["decrease", "lower", "sank", "loss", "drop", "fell", "underperform", "bankruptcy"]
            good = ["increase", "higher", "gain", "rise", "outperform"]

            if any(b in line for b in bad):
                file.write('1|'+ line + '\n')
            elif any(g in line for g in good):
                file.write('3|'+ line + '\n')
            else:
                try:
                    file.write('|'+ line + '\n')
                except:
                    print("\nThe following line could not be written:")
                    print(line + '\n')
    print("Finished scraping for " + url + '\n')

#---------------------------------------------------------------------------------------------------

    url = "https://finance.yahoo.com/topic/stock-market-news"
    print("Scraping: " + url)

    client = urlopen(url)
    html = client.read()
    client.close()

    page = soup(html, "html.parser")

    news = page.find_all("li", {"class":"js-stream-content"})
    #news = page.find_all("p", {"class": "Fz(14px) Lh(19px) Fz(13px)--sm1024 Lh(17px)--sm1024 LineClamp(2,38px) LineClamp(2,34px)--sm1024 M(0) C(#959595)"})
    for n in news:
        summary = n.find("h3")
        try:
            line = summary.get_text()
            first_sentence = line.split('.')[0]
            if check_if_news_exists(first_sentence, data) == False and "Final Round" not in line and "Akiko Fujita" not in line:
                print(line)
                bad = ["decrease", "lower", "sank", "loss", "drop", "fell", "underperform", "bankruptcy"]
                good = ["increase", "higher", "gain", "rise", "outperform"]

                if any(b in line for b in bad):
                    file.write('1|'+ line + '\n')
                elif any(g in line for g in good):
                    file.write('3|'+ line + '\n')
                else:
                    file.write('|'+ line + '\n')
        except:
            print("Line could not be foudn")


    print("Finished scraping for " + url + '\n')
    file.close()

def check_if_news_exists(first_sentence, data):
    for row in data:
        if first_sentence in row.split('|')[1]:
            return True
    return False

if __name__ == "__main__":
    get_news_for()
