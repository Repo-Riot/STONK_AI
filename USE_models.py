"""
Created on Sun Sep 11 21:55:14 2022

@author: User charlie 
"""


import time
import datetime
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from datetime import date
import pandas as pd
from IPython.display import display
import nasdaqdatalink 
import yfinance as yf
from yahoo_fin.stock_info import get_data
import fundamentalanalysis as fa
from stocksymbol import StockSymbol
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Stock_Symbol_API_KEY = "KEY_1"

start_of_program = time.time() 
company_returns = []

def print_company(company, Time_START, Time_end ):

    model = keras.models.load_model("100_6month_av")
    days_array, current_price = get_data_yf(company, Time_START, Time_end)
    prediction = load_models(days_array, current_price, model)

    print(f'{prediction}  %')

def scan(INDEX, Time_end, Time_START):
    """"""  
    company_tickers = []
    company_tickers_pass = []
    ss = StockSymbol("key_1")
    company_tickers_lists = ss.get_symbol_list(index=INDEX)

    model = keras.models.load_model("100_6month_median")
    
    for company_dic in company_tickers_lists:
        infomation = company_dic["symbol"]
        company_tickers.append(str(infomation))


    for company in company_tickers:
        
        try:
            info = get_data(company, start_date = Time_START, end_date = Time_end, index_as_date = True, interval = "1d")
            
            if len(info) > 100:
       
                company_tickers_pass.append(company)

        except:
            pass

    
    for company in company_tickers_pass:
        days_array, current_price = get_data_yf(company, Time_START, Time_end)

        prediction = load_models(days_array, current_price, model)
        
        company_returns.append((company, prediction))

    return(company_returns)    


def get_data_yf(company, Time_START, Time_end):
        
    #"Index(['open', 'high', 'low', 'close', 'adjclose', 'volume', 'ticker']"
    

    array = get_data(company, start_date = Time_START, end_date = Time_end, index_as_date = True, interval = "1d")
            
    array_adj = array["adjclose"]

    array = array_adj.to_numpy()
    arr = array[-100:]
    current_price = arr[-1]
    arr = arr.reshape(-1, 100)
    return(arr, current_price)    


def load_models(array, current_price, model):
    """"""

   
    price = model.predict(array, verbose=0)
    
    return(round(float((((price - current_price) / current_price) * 100)), 2))


def show_trades(prediction_list):
    """"""
    sorted_list = sorted(prediction_list, key=lambda x: x[1])

#select the five lowest tuples
    lowest_five = sorted_list[:10]

# select the five highest tuples
    highest_five = sorted_list[-10:]
    highest_five.reverse()

    print("Five Lowest :",lowest_five)
# Output : Five Lowest : [('item5', 1), ('item3', 2), ('item1', 3), ('item4', 4), ('item7', 5)]

    print("Five Highest :",highest_five)

Time_START = "2019-05-01"
Time_end = "2021-01-01"
Company = "SPY"

INDEX = "SPX"


#print_company(Company,  Time_START, Time_end)
prediction_list = scan(INDEX, Time_end, Time_START)
show_trades(prediction_list)
