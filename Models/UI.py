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

def ui(apikey, tickerlist, s_avg, v_avg):
    stock_avg, vol_avg = create_database(apikey, tickerlist, s_avg, v_avg)
    print(stock_avg)
    print(vol_avg)

    stock_avg, vol_avg = add_day(apikey, tickerlist, stock_avg, vol_avg)

    print(stock_avg)
    print(vol_avg)

if __name__ == "__main__":
    tickerlist = ast.literal_eval(sys.argv[2])
    stocklist = ast.literal_eval(sys.argv[3])
    vollist = ast.literal_eval(sys.argv[4])
    ui(sys.argv[1], tickerlist, stocklist, vollist)
