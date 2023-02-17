"""
Created on Sun Sep 11 21:55:14 2022

@author: User charlie 
"""

import time
import datetime
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
from datetime import date
import pandas as pd
from IPython.display import display
import nasdaqdatalink 
import yfinance as yf
from yahoo_fin.stock_info import get_data
import fundamentalanalysis as fa
from stocksymbol import StockSymbol
import tradingeconomics as te
import os 


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

start_of_program = time.time() 

pd.set_option("display.max_columns", None)

Stock_Symbol_API_KEY = "KEY_1"
Fundamental_analysis_API_KEY = "KEY_2"  #local api key
NASDAQ_DATA_LINK_API_KEY = "KEY_3" #local api key
nasdaqdatalink.ApiConfig.api_key = NASDAQ_DATA_LINK_API_KEY


def nan_counts(df):
  nan_counts = {}
  for col in df.columns:
    nan_count = df[col].isnull().sum()
    if nan_count.any():  # use any() to check if any values are True
      nan_counts[col] = (round(nan_count / len(df[col]), 2), nan_count)  # divide by length of column and round to two decimal places
  
  # Check for None values
  none_counts = {}
  for col in df.columns:
    none_count = df[col].eq(None).sum()
    if none_count.any():  # use any() to check if any values are True
      none_counts[col] = (round(none_count / len(df[col]), 2), none_count)  # divide by length of column and round to two decimal places
  
  # Merge the two dictionaries
  nan_counts.update(none_counts)
  
  return nan_counts

def drop_nan_index_rows(df):
  # Get the values of the first level of the multi-index
  index_level_0 = df.index.get_level_values(0)
  # Find rows where the index is null
  nan_index_rows = df[index_level_0.isnull()]
  # Drop these rows from the data frame
  df = df.drop(index=nan_index_rows.index)
  return df


def get_data_NASDAQ():
    
    #Index(['Open Interest', 'Dealer Longs', 'Dealer Shorts', 'Dealer Spreads',
    #      'Asset Manager Longs', 'Asset Manager Shorts', 'Asset Manager Spreads',
    #    'Leveraged Funds Longs', 'Leveraged Funds Shorts',
    #    'Leveraged Funds Spreads', 'Other Reportable Longs',
    #           'Other Reportable Shorts', 'Other Reportable Spreads',
    #     'Total Reportable Longs', 'Total Reportable Shorts',
    #    'Non Reportable Longs', 'Non Reportable Shorts']
     
    stock_dic = {'vix_all': '1170E1_FO_ALL', 
                 'oil_all': '067653_F_L_ALL' ,'wheat_all': '001601_F_L_ALL',
                 "Copper": "085691_F_L_ALL", "NZD" : "112741_F_ALL"           
                 }
    KEY = "vix_all"
    DATA_Index = 7
    commodity = stock_dic[KEY]
    
    data = nasdaqdatalink.get('CFTC/' + commodity)
    
    data_types = (data.axes[1]) #dates_from_pd_data_frame
    
    
    specifc_data = data_types[DATA_Index] 
    print("Types of information")
    print()
    print(data_types) #all_avaliable_infomation_types
    print()
    
    print()
    print("Type selected")
    print(specifc_data) # type selected
    print()
    
    specifc_data = data_types[DATA_Index] 
    data_numpy = data.to_numpy()
    ys_string = data_numpy[:, DATA_Index]
    print("Number of Data Points " + str(len(ys_string)))
 
    data_dates = []
    for dat in data.axes[0]: #data clean up
        (time) = str(dat)
        data_dates.append(time[:10])

    first_week = data_dates[0]
    last_week = data_dates[-1]
  
    xs = data_dates
    x_tags = []
    x_tickloc = list(np.linspace(0, len(ys_string) -1, num=11, dtype=int ))
    
    count = 0 
    for tag in data_dates:
        if count in x_tickloc:
            x_tags.append(tag)
        count += 1
    print("index ticks")
    print(x_tickloc)
    print()
    print("date of ticks")
    print(x_tags)
   
    plt.xticks(x_tickloc, labels=x_tags, rotation=90)    
    plt.title(KEY + " " + str(specifc_data) + " from " + str(first_week) + " to "  + str(last_week))
    plt.grid()
    plt.plot(xs, ys_string, color="green")
    plt.show()
    return(first_week,last_week)

def add_pct_change_columns(df, columns, interval):
    for column in columns:
        new_column_name = f"{column}_pct_change_{interval}"
        df[new_column_name] = df[column].pct_change(periods=interval)
        df.drop(columns=[column],inplace=True)
    return df
def replace_none_with_zero(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna(0)
    return df

def add_rolling_std_columns(df, columns, interval):
    for column in columns:
        new_column_name = f"{column}_rolling_std_{interval}"
        df[new_column_name] = df[column].rolling(window=interval).std()
    return df

def add_moving_average_columns(df, columns, interval):
    for column in columns:
        new_column_name = f"{column}_moving_avg_{interval}"
        df[new_column_name] = df[column].rolling(window=interval).mean()
    return df

def reindexs(df):
    columns_to_move = df.columns[154:166]
    columns_new_order = columns_to_move.tolist() + df.columns.tolist()[:154] + df.columns.tolist()[166:]
    df = df.reindex(columns=columns_new_order)
    return(df)

def add_cols(stonk_DF):
    

    stonk_DF["Market Cap made"] = stonk_DF["weightedAverageShsOut"] * stonk_DF["adjclose"]
    stonk_DF["PE made"] = stonk_DF["adjclose"] / stonk_DF["eps"] 
    stonk_DF["PE Market Cap made"] =  stonk_DF["eps"] / stonk_DF["Market Cap made"]
    stonk_DF["rev/adjclose made"] = stonk_DF["revenue"] / stonk_DF["adjclose"]
    stonk_DF["Market Cap/rev made"] =   stonk_DF["revenue"] / stonk_DF["Market Cap made"]
    stonk_DF["Market Cap/totalAssets made"] =  stonk_DF["totalAssets"] / stonk_DF["Market Cap made"]
    stonk_DF["totalAssetss/adjclose made"] =  stonk_DF["totalAssets"] / stonk_DF["adjclose"]

    
    stonk_DF["MACD"] = stonk_DF["MACD"].clip(lower=-20, upper=20)
    
    stonk_DF["cashAndCashEquivalents"] = stonk_DF["cashAndCashEquivalents"].clip(lower=0, upper=500000000000)
    
    stonk_DF["shortTermInvestments"] = stonk_DF["shortTermInvestments"].clip(lower=0, upper=2000000000)
    
    stonk_DF["inventory"] = stonk_DF["inventory"].clip(lower=0, upper=5000000000)
    
    stonk_DF["shortTermInvestments"] = stonk_DF["shortTermInvestments"].clip(lower=0, upper=2000000000)
        
    stonk_DF["accountPayables"] = stonk_DF["accountPayables"].clip(lower=0, upper=9000000000)
    
    stonk_DF["totalAssets"] = stonk_DF["totalAssets"].clip(lower=0, upper=350000000000)
    
    stonk_DF["shortTermDebt"] = stonk_DF["shortTermDebt"].clip(lower=0, upper=5000000000)
    
    stonk_DF["deferredRevenue"] = stonk_DF["deferredRevenue"].clip(lower=0, upper=1500000000)
    
    stonk_DF["totalDebt"] = stonk_DF["totalDebt"].clip(lower=0, upper=100000000000)
    
    stonk_DF["totalCurrentLiabilities"] = stonk_DF["totalCurrentLiabilities"].clip(lower=0, upper=100000000000)  
    
    stonk_DF["totalLiabilities"] = stonk_DF["totalLiabilities"].clip(lower=0, upper=400000000000) 
    
    stonk_DF["totalEquity"] = stonk_DF["totalEquity"].clip(lower=20000000, upper=50000000000)
    
    stonk_DF["totalStockholdersEquity"] = stonk_DF["totalStockholdersEquity"].clip(lower=10000000, upper=120000000000)
    
    stonk_DF["longTermDebt"] = stonk_DF["longTermDebt"].clip(lower=0, upper=50000000000)
    
    stonk_DF["depreciationAndAmortization"] = stonk_DF["depreciationAndAmortization"].clip(lower=0, upper=3000000000)
    
    stonk_DF["grossProfit"] = stonk_DF["grossProfit"].clip(lower=0, upper=50000000000)
    
    stonk_DF["netDebt"] = stonk_DF["netDebt"].clip(lower=-10000000000, upper=50000000000) 
    
    stonk_DF["stockBasedCompensation"] = stonk_DF["stockBasedCompensation"].clip(lower=0, upper=100000000)
    
    stonk_DF["changeInWorkingCapital"] = stonk_DF["changeInWorkingCapital"].clip(lower=-1000000000, upper=1000000000)
    
    stonk_DF["operatingIncomeRatio"] = stonk_DF["operatingIncomeRatio"].clip(lower=-1, upper=3)
       
    stonk_DF["totalCurrentAssets"] = stonk_DF["totalCurrentAssets"].clip(lower=0, upper=50000000000)
    
    stonk_DF["longTermInvestments"] = stonk_DF["longTermInvestments"].clip(lower=0, upper=5000000000)
    
    stonk_DF["grossProfitRatio"] = stonk_DF["grossProfitRatio"].clip(lower=0, upper=1)
    
    stonk_DF["ebitdaratio"] = stonk_DF["ebitdaratio"].clip(lower=-.5, upper=1.5)
    
    stonk_DF["netIncomeRatio"] = stonk_DF["netIncomeRatio"].clip(lower=-1, upper=1)
    
    stonk_DF["epsdiluted"] = stonk_DF["epsdiluted"].clip(lower=-10, upper=10)
    
    stonk_DF["PE Market Cap made"] = stonk_DF["PE Market Cap made"].clip(upper= 0.0000000009, lower=-0.000000002)
    
    stonk_DF["PE made"] = stonk_DF["PE made"].clip(upper=2000, lower=-1000)
    
    stonk_DF["volume"] = stonk_DF["volume"].clip(upper= 100000000)
    
    stonk_DF["PE Market Cap made"]  = stonk_DF["PE Market Cap made"].clip(lower=-100000)
    
    stonk_DF["eps"] = stonk_DF["eps"].clip(lower=-10)
    
    stonk_DF["revenue"] = stonk_DF["revenue"].clip(lower=0)
    
    stonk_DF["commonStockIssued"] = stonk_DF["commonStockIssued"].clip(lower=-1000000, upper= 200000000)
    
    stonk_DF["commonStockRepurchased"] = stonk_DF["commonStockRepurchased"].clip(lower=-1000000000, upper= 500000000)
    
    stonk_DF["operatingCashFlow"] = stonk_DF["operatingCashFlow"].clip(lower=10000000, upper= 5000000000)
    
    stonk_DF["capitalExpenditure"] = stonk_DF["capitalExpenditure"].clip(lower=-3000000000, upper= 50000000)
    
    stonk_DF["freeCashFlow"] = stonk_DF["freeCashFlow"].clip(lower=-1000000000, upper= 5000000000)
    
    stonk_DF["rev/adjclose  made"] = stonk_DF["rev/adjclose made"].clip(lower=-1000000000, upper= 5000000000)
    
    stonk_DF["Market Cap/rev  made"] = stonk_DF["Market Cap/rev made"].clip(lower=-1, upper= 5)
    
    stonk_DF["Market Cap/totalAssets  made"] = stonk_DF["Market Cap/totalAssets made"].clip(lower=-30, upper= 50)
   
    stonk_DF["totalAssetss/adjclose  made"] = stonk_DF["totalAssetss/adjclose made"].clip(lower=0, upper= 25000000000)
    
    stonk_DF = stonk_DF.reset_index()

    
    lag_ls = [30, 80]
    
    for lag_n in lag_ls:
        col_nm = 'return_day_lag' + str(lag_n)
        
        stonk_DF[col_nm] = stonk_DF['adjclose'] / stonk_DF.groupby('ticker')['adjclose'].shift(lag_n)
        stonk_DF[col_nm] = stonk_DF[col_nm].fillna(1)
        stonk_DF[col_nm] = (stonk_DF[col_nm]-1) * 100
       
    futures = [-10,-30, -90]
    
    for future in futures:
        col_nm = 'return_Future ' + str(abs((future)))
        
        stonk_DF[col_nm] = stonk_DF['adjclose'] / stonk_DF.groupby('ticker')['adjclose'].shift(future)
        stonk_DF[col_nm] = stonk_DF[col_nm].fillna(1)
        stonk_DF[col_nm] = (stonk_DF[col_nm]-1) * 100

    stonk_DF["day"] = pd.to_datetime(stonk_DF["day"])   
    stonk_DF = stonk_DF.set_index(['ticker', "day"])
    

    results_df = pd.DataFrame()
    results_df["adjclose"] = stonk_DF["adjclose"]
    results_df["return_Future 10"] = stonk_DF["return_Future 10"]
    results_df["return_Future 30"] = stonk_DF["return_Future 30"]
    results_df["return_Future 90"] = stonk_DF["return_Future 90"]    
    stonk_DF = stonk_DF.drop([ "return_Future 10", "return_Future 30", "return_Future 90" ], axis=1)

    

    
    stonk_DF = reindexs(stonk_DF)
    
    stonk_DF = normalize_dataframe(stonk_DF)

    #stonk_DF=(stonk_DF-stonk_DF.min())/(stonk_DF.max()-stonk_DF.min())


    
    return(stonk_DF, results_df)

      
def get_data_maco(Time_START, Time_end):
    
    Ticker = "AAPL"
 
    apple_frame = get_data(Ticker, start_date = Time_START, end_date = Time_end, index_as_date = True, interval = "1d")
    good_dates = list(apple_frame.axes[0])

    data = nasdaqdatalink.get("FRED/DFF", start_date=Time_START, end_date=Time_end)
    all_dates = list(data.axes[0])
    print(data.info())
                         
    values = data["Value"].tolist()
    
    print(len(good_dates),len(all_dates),len(values))
    count = 0 
    good_values = []
    all_dates_str = []
    for date in all_dates:
        value = str(date)[0:10]
        all_dates_str.append(value)
        
    for date in good_dates :
        if str(date)[0:10] in all_dates_str: 
            good_values.append(values[count])
            
        else: 
            good_values.append(good_values[-1])
            
        count += 1
    
           
    print(len(good_values))

def get_number_of_statments(Time_START, Time_end):
    """ gets the number of sheets required """
    NUMBER_STATMENTS_A_YEAR = 4

    year_differance = int(Time_end[:4]) - int(Time_START[:4]) 

    
    month_diffrenace = int(Time_end[5:7]) - int(Time_START[5:7])
    


    number_of_statments =  year_differance * NUMBER_STATMENTS_A_YEAR  + (month_diffrenace // 3) + 3
    print(f'time sample == {year_differance} years and {month_diffrenace} months')
    print(f"number of statments == {number_of_statments}")

    return(number_of_statments)

      
def get_data_yf(list_of_tickers, Time_START, Time_end):
        
    #"Index(['open', 'high', 'low', 'close', 'adjclose', 'volume', 'ticker']"
    start = time.time()
    
    Ticker = "AAPL"

    apple_frame = get_data(Ticker, start_date = Time_START, end_date = Time_end, index_as_date = True, interval = "1d")
    list(apple_frame.axes[0])
    Good_len = len(apple_frame)
    
    print("DATA_DAYS == " + str(Good_len))
    stock_dataframe = pd.DataFrame()
    count = 0 
    for ticker in list_of_tickers:
        try:
            info = get_data(ticker, start_date = Time_START, end_date = Time_end, index_as_date = True, interval = "1d")
            
            if len(info) == Good_len:
                count += 1
                
                stock_dataframe = pd.concat([stock_dataframe, info])
        except:
            pass
        
    end = time.time()
    print("NUMBER OF COMPANYS yf PASS == " + str(count))
    print("get_data_yf DONE TIME == " + str(round((end-start) /60, 2)) +" MINUTES")
    
    
    return(stock_dataframe)    



def get_SMA_feature(stock_dataframe, duration_ls = [10, 25, 100, 200]):
    tmp = stock_dataframe.copy()
    for n in duration_ls:
        col_nm = 'SMA' + str(n)
        tmp[col_nm] = tmp.groupby('ticker')['adjclose'].transform(lambda x: x.rolling(n, 1).mean())
    output_df = tmp
    print("SMA DONE")

    return output_df

## EMA
def get_EMA_feature(SMA, duration_ls = [10, 25, 100, 200]):
    tmp = SMA.copy()
    for n in duration_ls:
        col_nm = 'EMA' + str(n)
        tmp[col_nm] = tmp.groupby('ticker')['adjclose'].transform(lambda x: x.ewm(n, min_periods = 1).mean())
    output_df = tmp
    print("EMA DONE")
    return output_df

def get_MACD_feature(EMA):
    tmp = EMA.copy()
    EMA_12 = pd.Series(tmp.groupby('ticker')['adjclose'].transform(lambda x: x.ewm(12, min_periods = 1).mean()))
    EMA_26 = pd.Series(tmp.groupby('ticker')['adjclose'].transform(lambda x: x.ewm(26, min_periods = 1).mean()))
    tmp['MACD'] = pd.Series(EMA_12 - EMA_26)
    output_df = tmp
    print("MACD DONE")
    return output_df

def relative_strength_idx(sr, n=30):
    delta = sr.diff()
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi



def balance_sheet_statement(MACD, LIMIT_OF_fundermental_INFO):
    start = time.time()
    data = pd.DataFrame()
 
    list_of_companys = MACD['ticker'].unique()
    for company in list_of_companys:
        info = fa.balance_sheet_statement(company,Fundamental_analysis_API_KEY, period="quarter", limit=LIMIT_OF_fundermental_INFO)
        
        new_info = info.T
 
        if (len(new_info)) != LIMIT_OF_fundermental_INFO:
            print(company + " != balance_sheet_statement_LIMIT")  
            
        new_info["Company"] = str(company)
        data = pd.concat([data , new_info])    
    
        data.drop(columns='reportedCurrency', inplace=True)
        data.drop(columns='finalLink', inplace=True)
        data.drop(columns='link', inplace=True)
        data.drop(columns='minorityInterest', inplace=True)
        data.drop(columns='totalLiabilitiesAndStockholdersEquity', inplace=True)
        data.drop(columns='othertotalStockholdersEquity', inplace=True)
        data.drop(columns='accumulatedOtherComprehensiveIncomeLoss', inplace=True)
        data.drop(columns='preferredStock', inplace=True)
        data.drop(columns='cik', inplace=True)
        data.drop(columns='acceptedDate', inplace=True)
        data.drop(columns='calendarYear', inplace=True)
        data.drop(columns='period', inplace=True)
        data.drop(columns='otherCurrentAssets', inplace=True)
        data.drop(columns='propertyPlantEquipmentNet', inplace=True)
        data.drop(columns='goodwill', inplace=True)
        data.drop(columns='intangibleAssets', inplace=True)
        data.drop(columns='goodwillAndIntangibleAssets', inplace=True)
        data.drop(columns='otherNonCurrentAssets', inplace=True)
        data.drop(columns='otherAssets', inplace=True)
        data.drop(columns='taxPayables', inplace=True)
        data.drop(columns='otherCurrentLiabilities', inplace=True)
        data.drop(columns='deferredTaxLiabilitiesNonCurrent', inplace=True)
        data.drop(columns='otherNonCurrentLiabilities', inplace=True)
        data.drop(columns='capitalLeaseObligations', inplace=True)
        data.drop(columns='cashAndShortTermInvestments', inplace=True)
        data.drop(columns='taxAssets', inplace=True)
        data.drop(columns='totalNonCurrentAssets', inplace=True)
        data.drop(columns="totalNonCurrentLiabilities", inplace=True)
             
             
    columns_titles = ["Company","fillingDate", "cashAndCashEquivalents","shortTermInvestments",
                      "inventory","totalCurrentAssets","longTermInvestments","totalAssets",
                      "accountPayables","shortTermDebt","deferredRevenue","totalCurrentLiabilities","longTermDebt","deferredRevenueNonCurrent",
                      "otherLiabilities","totalLiabilities","commonStock","retainedEarnings","totalStockholdersEquity","totalEquity",
                      "totalLiabilitiesAndTotalEquity","totalInvestments","totalDebt","netDebt",]   
        
    
    data=data.reindex(columns=columns_titles)
    
    data = data.rename(columns = {'Company':'Company_BS', "fillingDate":"fillingDate_BH"})
    
    
    end = time.time()

    
    print("balance_sheet_statement Shape ==  " + str(data.shape))
    print("balance_sheet_statement DONE TIME == " + str(round((end-start) /60, 2)) +" MINUTES")
    
    return(data)
    
    
    #print(balance_sheet.info())
    #display(balance_sheet)
    
def income_statement(MACD, LIMIT_OF_fundermental_INFO):
    start = time.time()
    data = pd.DataFrame()
 
    list_of_companys = MACD['ticker'].unique()
    
    for company in list_of_companys:
        
        info = fa.income_statement(company,Fundamental_analysis_API_KEY, period="quarter", limit=LIMIT_OF_fundermental_INFO)
        
        new_info = info.T
        
        if (len(new_info)) != LIMIT_OF_fundermental_INFO:
            print(company + " != income_statement_LIMIT")
            
            
        new_info["Company"] = str(company)
        data = pd.concat([data , new_info])    
                
        
        data.drop(columns='reportedCurrency', inplace=True)
        data.drop(columns='cik', inplace=True)
        data.drop(columns='acceptedDate', inplace=True)
        data.drop(columns='calendarYear', inplace=True)
        data.drop(columns='period', inplace=True)
        data.drop(columns='link', inplace=True)
        data.drop(columns='finalLink', inplace=True)
        data.drop(columns='researchAndDevelopmentExpenses', inplace=True)
        data.drop(columns='sellingAndMarketingExpenses', inplace=True)
        data.drop(columns='sellingGeneralAndAdministrativeExpenses', inplace=True)
        data.drop(columns='otherExpenses', inplace=True)
        data.drop(columns='generalAndAdministrativeExpenses', inplace=True)
        data.drop(columns='interestIncome', inplace=True)
        data.drop(columns='interestExpense', inplace=True)
        data.drop(columns='incomeBeforeTax', inplace=True)  
        data.drop(columns='incomeBeforeTaxRatio', inplace=True)
        data.drop(columns='incomeTaxExpense', inplace=True)  
        data.drop(columns="totalOtherIncomeExpensesNet", inplace=True)  
        
        
        
    columns_titles = ["Company","fillingDate", "revenue","costOfRevenue","grossProfit",
                          "grossProfitRatio","operatingExpenses","costAndExpenses", "depreciationAndAmortization",
                          "ebitda","ebitdaratio","netIncome","operatingIncome","operatingIncomeRatio",
                          "netIncomeRatio","eps","epsdiluted","weightedAverageShsOut",
                          "weightedAverageShsOutDil"] 
            
            
    #Index: 1178 entries, 2022-06 to 2012-09
    data=data.reindex(columns=columns_titles)
    data = data.rename(columns = {'Company':'Company_IS', "fillingDate":"fillingDate_IS"})
    
    end = time.time()

    print("income_statement Shape ==  " + str(data.shape))
    print("income_statement DONE TIME == " + str(round((end-start) /60, 2)) +" MINUTES")
    
    
    return(data)
  
def cash_flow_statement(MACD, LIMIT_OF_fundermental_INFO):
    start = time.time()
    data = pd.DataFrame()
  
    list_of_companys = MACD['ticker'].unique()
    
    for company in list_of_companys:
        
        info = fa.cash_flow_statement(company,Fundamental_analysis_API_KEY, period="quarter", limit=LIMIT_OF_fundermental_INFO)
        
        new_info = info.T
        
        if (len(new_info)) != LIMIT_OF_fundermental_INFO:
            print(company + " != cash flow statement_LIMIT")
                        
        new_info["Company"] = str(company)
        data = pd.concat([data , new_info])   
        
        data.drop(columns='reportedCurrency', inplace=True)
        data.drop(columns='cik', inplace=True)
        data.drop(columns='acceptedDate', inplace=True)
        data.drop(columns='calendarYear', inplace=True)        
        data.drop(columns='period', inplace=True)
        data.drop(columns='netIncome', inplace=True)
        data.drop(columns='depreciationAndAmortization', inplace=True)
        data.drop(columns='deferredIncomeTax', inplace=True)
        data.drop(columns='accountsReceivables', inplace=True)
        data.drop(columns='inventory', inplace=True)  
        data.drop(columns='accountsPayables', inplace=True)
        data.drop(columns='otherWorkingCapital', inplace=True)
        data.drop(columns='otherNonCashItems', inplace=True)
        data.drop(columns='netCashProvidedByOperatingActivities', inplace=True)
        data.drop(columns='investmentsInPropertyPlantAndEquipment', inplace=True)
        data.drop(columns='purchasesOfInvestments', inplace=True)
        data.drop(columns='salesMaturitiesOfInvestments', inplace=True) 
        data.drop(columns='otherInvestingActivites', inplace=True) 
        data.drop(columns='netCashUsedForInvestingActivites', inplace=True)
        data.drop(columns='otherFinancingActivites', inplace=True) 
        data.drop(columns='netCashUsedProvidedByFinancingActivities', inplace=True)
        data.drop(columns='effectOfForexChangesOnCash', inplace=True)
        data.drop(columns='netChangeInCash', inplace=True)
        data.drop(columns='cashAtBeginningOfPeriod', inplace=True)
        data.drop(columns='link', inplace=True)
        data.drop(columns='finalLink', inplace=True)  
        
    columns_titles = ["Company","fillingDate", "stockBasedCompensation","changeInWorkingCapital","acquisitionsNet",
                      "debtRepayment","commonStockIssued","commonStockRepurchased","operatingCashFlow","capitalExpenditure","freeCashFlow"] 
                
    data=data.reindex(columns=columns_titles) 
    data = data.rename(columns = {'Company':'Company_CF', "fillingDate":"fillingDate_CF"})
    end = time.time()  
    

    print("cash flow statement Shape ==  " + str(data.shape))
    print("cash flow statement DONE TIME == " + str(round((end-start) /60, 2)) +" MINUTES") 
    
    return(data)
    
def key_metrics(MACD, LIMIT_OF_fundermental_INFO):
    start = time.time()
    data = pd.DataFrame()
  
    list_of_companys = MACD['ticker'].unique()
    
    for company in list_of_companys:
        
        info = fa.key_metrics(company,Fundamental_analysis_API_KEY, period="quarter", limit=LIMIT_OF_fundermental_INFO)
        
        new_info = info.T
        
        if (len(new_info)) != LIMIT_OF_fundermental_INFO:
            print(company + " != key_metrics_LIMIT")
                        
        new_info["Company"] = str(company)
        data = pd.concat([data , new_info]) 
        
    
    data.drop(columns='period', inplace=True)
    
       
                        
    
    
    data = data.reset_index(drop=False, col_level=1, col_fill="index")
    
    data['index'] = pd.to_datetime(data['index'])
    data['date'] = data['index'].dt.strftime('%Y-%m-%d')
 


    
    end = time.time()   

        
    print("key_metrics Shape ==  " + str(data.shape))
    print("key_metrics DONE TIME == " + str(round((end-start) /60, 2)) +" MINUTES") 
    
    return(data)
    
def financial_ratios(MACD, LIMIT_OF_fundermental_INFO):
    
    start = time.time()
    data = pd.DataFrame()
  
    list_of_companys = MACD['ticker'].unique()
    
    for company in list_of_companys:
        
        info = fa.financial_ratios(company,Fundamental_analysis_API_KEY, period="quarter", limit=LIMIT_OF_fundermental_INFO)
        
        new_info = info.T
        
        if (len(new_info)) != LIMIT_OF_fundermental_INFO:
            print(company + " != financial_ratios_LIMIT")
                        
        new_info["Company"] = str(company)
        data = pd.concat([data , new_info]) 
        

        
    data.drop(columns='dividendYield', inplace=True)
    data.drop(columns='period', inplace=True)

    data = data.reset_index(drop=False, col_level=1, col_fill="index")
    
    data['index'] = pd.to_datetime(data['index'])
    data['date'] = data['index'].dt.strftime('%Y-%m-%d')
 



    

    end = time.time()   
      
    print("financial_ratios Shape ==  " + str(data.shape))
    print("financial_ratios DONE TIME == " + str(round((end-start) /60, 2)) +" MINUTES")   
    return(data)
    
def financial_growth(MACD, LIMIT_OF_fundermental_INFO):
    start = time.time()
    data = pd.DataFrame()
  
    list_of_companys = MACD['ticker'].unique()
    
    for company in list_of_companys:
        
        info = fa.financial_statement_growth(company,Fundamental_analysis_API_KEY, period="quarter", limit=LIMIT_OF_fundermental_INFO)
        
        new_info = info.T
        
        if (len(new_info)) != LIMIT_OF_fundermental_INFO:
            print(company + " != financial_growth_LIMIT")
                        
        new_info["Company"] = str(company)
        data = pd.concat([data , new_info]) 
    

        
        
    end = time.time()   
       
    print("financial_growth ==  " + str(data.shape))   
    print("financial_growth DONE TIME == " + str(round((end-start) /60, 2)) + " MINUTES")   
    
    return(data)
    
def get_tickers(LIMIT_OF_fundermental_INFO,INDEX):
    """gets inital ticker list then checks that have the correct anount of 
    balance sheets and income statments """
    start = time.time()
    company_tickers = []
    company_tickers_pass = []
    ss = StockSymbol("KEY_1")
    company_tickers_lists = ss.get_symbol_list(index=INDEX)
    for company_dic in company_tickers_lists:
        infomation = company_dic["symbol"]
        company_tickers.append(str(infomation))
    
    print("Number of TICKERS from get_symbol_lists == " + str(len(company_tickers_lists)))
    
    for company in company_tickers:
        BS = fa.balance_sheet_statement(company,Fundamental_analysis_API_KEY, period="quarter", limit=LIMIT_OF_fundermental_INFO)
        BST = BS.T
        BS_LEN = len(BST)
        
        IS = fa.income_statement(company,Fundamental_analysis_API_KEY, period="quarter", limit=LIMIT_OF_fundermental_INFO)
        IST = IS.T
        IS_LEN = len(IST)
        
        CF = fa.cash_flow_statement(company,Fundamental_analysis_API_KEY, period="quarter", limit=LIMIT_OF_fundermental_INFO)
        CFT = CF.T
        CF_LEN = len(CFT)
        
        KM = fa.financial_ratios(company,Fundamental_analysis_API_KEY, period="quarter", limit=LIMIT_OF_fundermental_INFO)
        KMT = KM.T
        KM_LEN = len(KMT)

        if IS_LEN == LIMIT_OF_fundermental_INFO and BS_LEN == LIMIT_OF_fundermental_INFO and CF_LEN == LIMIT_OF_fundermental_INFO and KM_LEN == LIMIT_OF_fundermental_INFO:   
            company_tickers_pass.append(company)
            
    end = time.time()
    print("Number of TICKERS from get_get_symbol_list to pass fundermental == " + str(len(company_tickers_pass)))
    print("get_tickers DONE TIME == " + str(round((end-start) /60, 2)) +" MINUTES")
    return(company_tickers_pass)





#MACD.to_hdf("DAILY " + INDEX + " START = " + Time_START + "   END = " + Time_end,key="MACD", index = False, encoding='utf-8')
#Fundermental_df.to_hdf("Fundermental " + INDEX + " START = " + Time_START + "   END = " + Time_end,key="Fundermental_df", index = False, encoding='utf-8')

#get_data_maco(Time_START, Time_end)
#get_data_fundermental()
#get_data_NASDAQ()
#get_data_yf_finance()




def data_days_array(stonk_META):
    
    list_of_companys = stonk_META['ticker'].unique()
    first_company = list_of_companys[0]   
    company_df =  stonk_META[stonk_META['ticker'] == first_company]
    
    all_dates_str = []
    the_list = list(company_df.axes[0])
    for date in the_list:
        value = str(date)[0:10]
        all_dates_str.append(value) 
        
    return(all_dates_str, list_of_companys)

def check_index(df):
  index_level_0 = df.index.get_level_values(0)
  if index_level_0.isnull().any():
    return index_level_0.isnull().sum()
  else:
    return 0

def check_index_single(df):
  index_level_0 = df.index.get_level_values(0)
  if index_level_0.isnull().any():
    return index_level_0.isnull().sum()
  else:
    return 0

def filter_df(df, column, values_to_remove):
  return df[~df[column].isin(values_to_remove)]

def get_most_recent_date(date, date_list):
  # Convert the input date to a datetime object
  reference_date = datetime.datetime.strptime(date, '%Y-%m-%d')

  # Initialize a variable to store the most recent date
  most_recent_date = "1950-01-01"
  most_recent_date = datetime.datetime.strptime(most_recent_date, '%Y-%m-%d')
  # Iterate through the list of dates
  
  for d in date_list:
    # Convert the current date to a datetime object
        
    if d is not None and d != "":
    
        current_date = datetime.datetime.strptime(d, '%Y-%m-%d')

    # Check if the current date is more recent than the most recent date
    # and is also earlier than or equal to the reference date
        if current_date <= reference_date and current_date > most_recent_date:
      # If the current date meets these conditions, update the most recent date
            most_recent_date = current_date

  # Return the most recent date
  return most_recent_date

def replace_none_with_zero(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna(0)
    return df

def remove_duplicate_columns(df):
    # Get the list of column names
    columns = df.columns
    # Find any duplicate column names and store them in a set
    duplicate_columns = set([x for x in columns if columns.count(x) > 1])
    # Drop the duplicate columns
    df = df.drop(columns=list(duplicate_columns))
    return df
def convert_columns_to_float(df, columns):
  for col in columns:
    df[col] = df[col].astype(float)
  return df

def remove_values(list1, list2):
    # Create a set of the values in list2
    set2 = set(list2)
    # Iterate over the elements in list1 and add them to a new list if they are not in set2
    new_list = [x for x in list1 if x not in set2]
    # Return the new list
    return new_list

def add_vix_sentiment(df, good_companys, Time_START, Time_end):

    good_companys = list(filter(lambda x: x is not None and not pd.isnull(x), good_companys))
    final_df = pd.DataFrame()
    info = get_data("^VIX", start_date = Time_START, end_date = Time_end, index_as_date = True, interval = "1d")
    info["adjclose"] = info["adjclose"]
    info["adjclose 30 average"] = info["adjclose"].rolling(window=30, min_periods=1).mean()
    info = info[["adjclose", "adjclose 30 average"]]
    info["adjclose 30 average change"] = info["adjclose 30 average"].pct_change(periods=30).fillna(0)
    
    info = info.dropna()
    info.reset_index()
    print(len(info))
    for company in good_companys:
        company_df = df[df["Company_IS"] == company]
        
        company_df = company_df.assign(vix_30_av=info["adjclose 30 average"].to_list())
        company_df = company_df.assign(vix_30_av_change=info["adjclose 30 average change"].to_list())

        final_df = pd.concat([final_df, company_df])

    return(final_df)


def maco_data(df, good_companys, Time_START, Time_end):
    """""" 
    # USD vs trade partners 
    good_companys = list(filter(lambda x: x is not None and not pd.isnull(x), good_companys))
    final_df = pd.DataFrame()
    info = get_data("^GSPC", start_date = Time_START, end_date = Time_end, index_as_date = True, interval = "1d")


    info["adjclose"]  = info["adjclose"]
    info["adjclose 30 average"] = info["adjclose"].rolling(window=30, min_periods=1).mean()
    info = info[["adjclose", "adjclose 30 average"]]
    info["adjclose 30 average change"] = info["adjclose 30 average"].pct_change(periods=30).fillna(0)
    info.reset_index()


    for company in good_companys:
        company_df = df[df["Company_IS"] == company]
        #cash
        company_df = company_df.assign(GSPC_30_av=info["adjclose 30 average"].to_list())
        company_df = company_df.assign(GSPC_change=info["adjclose 30 average change"].to_list())
        
        final_df = pd.concat([final_df, company_df])

    return(final_df)

def add_days(df, date_list, good_companys):

    good_companys = list(filter(lambda x: x is not None and not pd.isnull(x), good_companys))
    final_df = pd.DataFrame()
    for company in good_companys:
        
       
        
        company_df = df[df["Company_IS"] == company]
        
        company_df = company_df.assign(day=date_list)
        
        final_df = pd.concat([final_df, company_df])
    
    return(final_df)


def new_balance_sheet_to_len(all_dates_str, Balance_sheet, list_of_companys, bad_companys):
    """does this properly """

    start = time.time()
    
    
    BC_DF = pd.DataFrame()
    
    list_of_companys = list(filter(lambda x: x is not None and not pd.isnull(x), list_of_companys))
    
    for company in list_of_companys:
        
        Comp_BC =pd.DataFrame()
       
        
        company_df = Balance_sheet[Balance_sheet["Company_BS"] == company]
 
        


        company_fill_days = company_df["fillingDate_BH"].to_list()
        
    
        for day in all_dates_str:
            

            fill_date_used = get_most_recent_date(day, company_fill_days)
            fill_date_used = str(fill_date_used)
            fill_date_used = fill_date_used[0:10]
            row = company_df[(company_df["fillingDate_BH"]) == (fill_date_used)] ### not working 
            
            Comp_BC =  pd.concat([Comp_BC, row])
        
        if len(Comp_BC) > len(all_dates_str) or len(Comp_BC) < len(all_dates_str):
            bad_companys.append(company)
        
        
        
        
        BC_DF = pd.concat([BC_DF, Comp_BC])

    
    BC_DF = filter_df(BC_DF, "Company_BS", bad_companys)
    


    end = time.time()
    print(bad_companys)
    print("balance_sheet_to_len DONE TIME == " + str(round((end-start) /60, 2)) + " MINUTES")
    return(BC_DF, bad_companys)
   
def new_income_statment(all_dates_str, income_statment, list_of_companys, bad_companys):
    
    start = time.time()
    
    
    IS_DF = pd.DataFrame()
    
    list_of_companys = list(filter(lambda x: x is not None and not pd.isnull(x), list_of_companys))
    
    for company in list_of_companys:
        
        Comp_BC =pd.DataFrame()
       
        
        company_df = income_statment[income_statment["Company_IS"] == company]

       
        
        company_fill_days = company_df["fillingDate_IS"].to_list()
        
    
        for day in all_dates_str:
            
            fill_date_used = get_most_recent_date(day, company_fill_days)
            fill_date_used = str(fill_date_used)
            fill_date_used = fill_date_used[0:10]
            row = company_df[(company_df["fillingDate_IS"]) == (fill_date_used)] 
            Comp_BC =  pd.concat([Comp_BC, row])
            
        if (len(Comp_BC) > len(all_dates_str) or len(Comp_BC) < len(all_dates_str)) and company not in bad_companys:
            bad_companys.append(company)
        
        IS_DF = pd.concat([IS_DF, Comp_BC])

    IS_DF = filter_df(IS_DF, "Company_IS", bad_companys)

    end = time.time()
    print(bad_companys)
    print("income_sheet_to_len DONE TIME == " + str(round((end-start) /60, 2)) + " MINUTES")
    return(IS_DF, bad_companys)



def new_cash_flow_statment(all_dates_str, cash_flow_statment,  list_of_companys, bad_companys):

    start = time.time()
    
    
    CF_DF = pd.DataFrame()
    
    list_of_companys = list(filter(lambda x: x is not None and not pd.isnull(x), list_of_companys))
    
    for company in list_of_companys:
        
        Comp_BC =pd.DataFrame()
       
        
        company_df = cash_flow_statment[cash_flow_statment["Company_CF"] == company]

       
        
        company_fill_days = company_df["fillingDate_CF"].to_list()
        
    
        for day in all_dates_str:
            
            fill_date_used = get_most_recent_date(day, company_fill_days)
            fill_date_used = str(fill_date_used)
            fill_date_used = fill_date_used[0:10]
            row = company_df[(company_df["fillingDate_CF"]) == (fill_date_used)]  
            
            Comp_BC =  pd.concat([Comp_BC, row])

        if (len(Comp_BC) > len(all_dates_str) or len(Comp_BC) < len(all_dates_str)) and company not in bad_companys:
            bad_companys.append(company)   

        
        CF_DF = pd.concat([CF_DF, Comp_BC])


    CF_DF = filter_df(CF_DF, "Company_CF", bad_companys)

    end = time.time()
    print(bad_companys)
    print("cash flow sheet_to_len DONE TIME == " + str(round((end-start) /60, 2)) + " MINUTES")
    return(CF_DF, bad_companys)
    
def new_key_metrics_to_len(all_dates_str, Key_metircs, list_of_companys, bad_companys):

    start = time.time()
    
    
    KM_DF = pd.DataFrame()
    
    list_of_companys = list(filter(lambda x: x is not None and not pd.isnull(x), list_of_companys))
    
    for company in list_of_companys:
        
        Comp_BC =pd.DataFrame()
       
        
        company_df = Key_metircs[Key_metircs["Company"] == company]

       
        
        company_fill_days = company_df["date"].to_list()
        
    
        for day in all_dates_str:
            
            fill_date_used = get_most_recent_date(day, company_fill_days)
            fill_date_used = str(fill_date_used)
            fill_date_used = fill_date_used[0:10]
            row = company_df[(company_df["date"]) == (fill_date_used)] 
            Comp_BC =  pd.concat([Comp_BC, row])
            
        if (len(Comp_BC) > len(all_dates_str) or len(Comp_BC) < len(all_dates_str)) and company not in bad_companys:
            bad_companys.append(company)
        
        KM_DF = pd.concat([KM_DF, Comp_BC])

    KM_DF = filter_df(KM_DF, "Company", bad_companys)

    end = time.time()
    print(bad_companys)
    print("KEY METRICS to_len DONE TIME == " + str(round((end-start) /60, 2)) + " MINUTES")
    return(KM_DF, bad_companys)

def new_ratios_to_len(all_dates_str, finance_ratio, list_of_companys, bad_companys):
    start = time.time()
    
    
    FK_DF = pd.DataFrame()
    
    list_of_companys = list(filter(lambda x: x is not None and not pd.isnull(x), list_of_companys))

    for company in list_of_companys:
        
        Comp_BC =pd.DataFrame()
       
        
        company_df = finance_ratio[finance_ratio["Company"] == company]

       
        
        company_fill_days = company_df["date"].to_list()
        
    
        for day in all_dates_str:
            
            fill_date_used = get_most_recent_date(day, company_fill_days)
            fill_date_used = str(fill_date_used)
            fill_date_used = fill_date_used[0:10]
            row = company_df[(company_df["date"]) == (fill_date_used)] 
            Comp_BC =  pd.concat([Comp_BC, row])
            
        if (len(Comp_BC) > len(all_dates_str) or len(Comp_BC) < len(all_dates_str)) and company not in bad_companys:
            bad_companys.append(company)
        
        FK_DF = pd.concat([FK_DF, Comp_BC])

    FK_DF = filter_df(FK_DF, "Company", bad_companys)
    
    end = time.time()
    print(bad_companys)
    print("Ratios_to_len DONE TIME == " + str(round((end-start) /60, 2)) + " MINUTES")
    return(FK_DF, bad_companys)

def merge_clean(stonk_META, Balance_shee, Income_shee, Cash_flow, key_metric, ratios ):
    """""" 
    print(f'nans in stonk_META = {nan_counts(stonk_META)}')
    print(f'nans in BS_DF = {nan_counts(Balance_shee)}')
    print(f'nans in IS_DF = {nan_counts(Income_shee)}')
    print(f'nans in CF_DF = {nan_counts(Cash_flow)}')
    print(f'nans in KM_DF = {nan_counts(key_metric)}')
    print(f'nans in FR_DF = {nan_counts(ratios)}')


def normalize_dataframe(df):

    df1 = pd.read_hdf('TRAIN_DF')

    mean = df1.mean()
    std = df1.std()

    normalized_df = (df - mean) / std
    
    return normalized_df

def stripdataframe(df):
    """"""""

    unique_names = df.index.get_level_values(0).unique().tolist()

    row_dataframe = pd.DataFrame()
    for company in unique_names:

        filtered_df = df.loc[company]

        last_row = filtered_df.tail(1)

        row_dataframe = pd.concat([row_dataframe, last_row])

    row_dataframe = row_dataframe.reset_index(drop=True)

    print(row_dataframe.info())

    

    arrays = row_dataframe.values.tolist()
    
    num_array = np.array([np.array(xi) for xi in arrays])
    print(num_array.shape)
    

    
    return(num_array, unique_names, row_dataframe )



def get_current_prices(Results_DF):

    current_prices = []

    Results_DF = Results_DF[["adjclose"]]

    unique_names = df.index.get_level_values(0).unique().tolist()
    
    for company in unique_names:

        filtered_df = df.loc[company]
        last_row = filtered_df.tail(1)
        current_prices.append(last_row['adjclose'].values[0])

    return(current_prices)



    
def add_bonds(all_date_str, Time_START, Time_end, df, good_companys):

    bond_data_13 =  get_data("^IRX", start_date = Time_START, end_date = Time_end, index_as_date = True, interval = "1d")
    bond_data_13 = bond_data_13[["adjclose"]]
    bond_data_13 = bond_data_13.rename(columns={'adjclose': '13_week_bond'})
    bond_data_13 =  bond_data_13.dropna()
    list_data = bond_data_13.index.to_list()
    bond_data_13 = bond_data_13.reset_index(level=0, drop=False)

    string_list = [date.strftime('%Y-%m-%d') for date in list_data]
    

    info_10 =  get_data("^TNX", start_date = Time_START, end_date = Time_end, index_as_date = True, interval = "1d")
    info_10 = info_10[["adjclose"]]
    info_10 = info_10.rename(columns={'adjclose': '10_year_bond'})
    info_10 =  info_10.dropna()
    info_10 = info_10.reset_index(level=0, drop=False)

    info_30 =  get_data( "^TYX", start_date = Time_START, end_date = Time_end, index_as_date = True, interval = "1d")
    info_30 = info_30[["adjclose"]]
    info_30 = info_30.rename(columns={'adjclose': '30_year_bond'})
    info_30 = info_30.dropna()
    info_30 = info_30.reset_index(level=0, drop=False)

    vix = get_data( "^VIX", start_date = Time_START, end_date = Time_end, index_as_date = True, interval = "1d")
    vix = vix[["adjclose"]]
    vix = vix.rename(columns={'adjclose': 'VIX'})
    vix = vix.dropna()
    vix = vix.reset_index(level=0, drop=False)
 
    vix_to_len = pd.DataFrame()
    bond_13_to_len = pd.DataFrame()
    bond_10_to_len = pd.DataFrame()
    bond_30_to_len = pd.DataFrame()

    for date in all_date_str:

        fill_date_used = get_most_recent_date(date, string_list)
        fill_date_used = str(fill_date_used)
        fill_date_used = fill_date_used[0:10]

        row = bond_data_13[(bond_data_13["index"]) == (fill_date_used)] 
        bond_13_to_len =  pd.concat([bond_13_to_len, row])
    
        row_10 = info_10[(info_10["index"]) == (fill_date_used)]
        bond_10_to_len =  pd.concat([bond_10_to_len, row_10])
        
        row_30 = info_30[(info_30["index"]) == (fill_date_used)]
        bond_30_to_len =  pd.concat([bond_30_to_len, row_30])

        row_vix = vix[(vix["index"]) == (fill_date_used)]
        vix_to_len = pd.concat([vix_to_len, row_vix]) 

    
    bond_30_to_len = bond_30_to_len.drop(columns=['index'])
    bond_10_to_len = bond_10_to_len.drop(columns=['index'])
    bond_13_to_len = bond_13_to_len.drop(columns=['index'])
    vix_to_len = vix_to_len.drop(columns=['index'])

  
    bond_30_to_len = bond_30_to_len.reset_index(drop=True)
    vix_to_len = vix_to_len.reset_index(drop=True)
    bond_13_to_len = bond_13_to_len.reset_index(drop=True)
    bond_10_to_len = bond_10_to_len.reset_index(drop=True)
    
    all_bond_df = pd.concat([bond_13_to_len, bond_10_to_len, bond_30_to_len, vix_to_len], axis=1, ignore_index=False)


    all_bond_df["VIX_30_av"] = all_bond_df["VIX"].rolling(window=30, min_periods=1).mean()
    all_bond_df['13_week_bond_30_av'] =  all_bond_df["13_week_bond"].rolling(window=30, min_periods=1).mean()
    all_bond_df['10_year_bond_30_av'] =  all_bond_df["10_year_bond"].rolling(window=30, min_periods=1).mean()
    all_bond_df['30_year_bond_30_av'] =  all_bond_df["30_year_bond"].rolling(window=30, min_periods=1).mean()

    all_bond_df["13_week_bond_30_average change"] =  all_bond_df['13_week_bond_30_av'].pct_change(periods=30).fillna(0)
    all_bond_df["10_year_bond_30_average change"] =  all_bond_df['10_year_bond_30_av'].pct_change(periods=30).fillna(0)
    all_bond_df["30_year_bond_30_average change"] =  all_bond_df['30_year_bond_30_av'].pct_change(periods=30).fillna(0)
    all_bond_df["VIX_30_average_change"]  =  all_bond_df['VIX_30_av'].pct_change(periods=30).fillna(0)


    all_bond_df["yeild_curve_10"] = all_bond_df['10_year_bond_30_av']  - all_bond_df['13_week_bond_30_av']
    all_bond_df["yeild_curve_30"] = all_bond_df['30_year_bond_30_av']  - all_bond_df['13_week_bond_30_av']

    
    all_bond_df = all_bond_df.drop(['VIX', '13_week_bond', '10_year_bond', '30_year_bond'], axis=1)


    final_df = pd.DataFrame()

    for company in good_companys:

        company_df = df[df["Company_IS"] == company]
        company_df = company_df.assign( week_13_bond_30_av=all_bond_df['13_week_bond_30_av'].to_list())
        company_df = company_df.assign(year_10_bond_30_av=all_bond_df['10_year_bond_30_av'].to_list())
        company_df = company_df.assign(year_30_bond_30_av=all_bond_df['30_year_bond_30_av'].to_list())

        company_df = company_df.assign(week_13_bond_chnage=all_bond_df['13_week_bond_30_average change'].to_list())
        company_df = company_df.assign(year_10_bond_chnage=all_bond_df['10_year_bond_30_average change'].to_list())
        company_df = company_df.assign(year_30_bond_chnage=all_bond_df['30_year_bond_30_average change'].to_list())

        company_df = company_df.assign(yeild_curve_10=all_bond_df['yeild_curve_10'].to_list())
        company_df = company_df.assign(yeild_curve_30=all_bond_df['yeild_curve_30'].to_list())

        company_df = company_df.assign(VIX_30_av=all_bond_df['VIX_30_av'].to_list())
        company_df = company_df.assign(VIX_30_average_change=all_bond_df['VIX_30_average_change'].to_list())
        
        final_df = pd.concat([final_df, company_df])

    return(final_df)


def finish_data_frame(Time_START, Time_end, INDEX):
    numbers_of_statments = get_number_of_statments(Time_START, Time_end) # gets the amount of statments for required period 
    LIMIT_OF_fundermental_INFO = numbers_of_statments

    list_of_tickers = get_tickers(LIMIT_OF_fundermental_INFO, INDEX)
    stock_dataframe = get_data_yf(list_of_tickers, Time_START, Time_end)


    SMA = get_SMA_feature(stock_dataframe, duration_ls = [10, 25, 100, 200])
    EMA = get_EMA_feature(SMA, duration_ls = [10, 25, 100, 200])
    MACD = get_MACD_feature(EMA)
    MACD['RSI'] = MACD.groupby('ticker')['adjclose'].transform(lambda x: relative_strength_idx(x, n = 10))
    MACD["RSI"] = MACD["RSI"].replace(np.nan, 50)
    stonk_META = MACD   


    Balance_sheet = balance_sheet_statement(MACD, LIMIT_OF_fundermental_INFO)

    Income_statement = income_statement(MACD, LIMIT_OF_fundermental_INFO)

    Cash_flow_statement = cash_flow_statement(MACD, LIMIT_OF_fundermental_INFO)  

    Key_metrics = key_metrics(MACD, LIMIT_OF_fundermental_INFO) # missing data

    Financial_ratios = financial_ratios(MACD, LIMIT_OF_fundermental_INFO) # missing data

    #Growth_quarterly = financial_growth(MACD, LIMIT_OF_fundermental_INFO) very poor data

    all_dates_str, list_of_companys = data_days_array(stonk_META)

    bad_companys = []

    Balance_shee, bad_companys = new_balance_sheet_to_len(all_dates_str, Balance_sheet, list_of_companys, bad_companys)
    Balance_shee = Balance_shee.reset_index()

    Income_shee, bad_companys = new_income_statment(all_dates_str, Income_statement, list_of_companys, bad_companys)
    Income_shee = Income_shee.reset_index()

    Cash_flow, bad_companys = new_cash_flow_statment(all_dates_str, Cash_flow_statement, list_of_companys, bad_companys)
    Cash_flow = Cash_flow.reset_index()

    key_metric, bad_companys = new_key_metrics_to_len(all_dates_str, Key_metrics, list_of_companys, bad_companys)
    print(nan_counts(key_metric))
    key_metric = replace_none_with_zero(key_metric)
    key_metric = key_metric.reset_index()

    ratios, bad_companys = new_ratios_to_len(all_dates_str, Financial_ratios, list_of_companys, bad_companys)
    print(nan_counts(ratios))
    ratios = replace_none_with_zero(ratios)
    ratios = ratios.reset_index()

    print(bad_companys)
    # FINAl_DF = merge_clean()

    print(f'nans in stonk_META = {nan_counts(stonk_META)}')
    print(f'nans in BS_DF = {nan_counts(Balance_shee)}')
    print(f'nans in IS_DF = {nan_counts(Income_shee)}')
    print(f'nans in CF_DF = {nan_counts(Cash_flow)}')
    print(f'nans in KM_DF = {nan_counts(key_metric)}')
    print(f'nans in FR_DF = {nan_counts(ratios)}')



    stonk_META = stonk_META.reset_index()
    FINAl_DF = pd.concat([stonk_META, Balance_shee, Income_shee, Cash_flow, key_metric, ratios], axis=1)

    good_companys = remove_values(list_of_companys, bad_companys)
    FINAl_DF = FINAl_DF.drop(labels=FINAl_DF.columns[FINAl_DF.columns.duplicated()], axis=1)
    FINAl_DF = add_days(FINAl_DF, all_dates_str, good_companys)
    # FINAl_DF = add_vix_sentiment(FINAl_DF, good_companys, Time_START, Time_end)
    FINAl_DF = maco_data(FINAl_DF, good_companys, Time_START, Time_end)
    FINAl_DF  = add_bonds(all_dates_str, Time_START, Time_end, FINAl_DF, good_companys)

    FINAl_DF.drop(columns=['Company_BS', 'fillingDate_BH', 'Company_IS', 'fillingDate_IS', 'Company_CF', 'fillingDate_CF'], inplace=True)
    FINAl_DF.reset_index(drop=True, inplace=True)
    FINAl_DF = FINAl_DF.set_index(['ticker', 'day'])
    object_columns = [col for col in FINAl_DF.columns if FINAl_DF[col].dtype == 'object']
    FINAl_DF = convert_columns_to_float(FINAl_DF, object_columns)

    print(check_index(FINAl_DF)) 

    FINAl_DF = drop_nan_index_rows(FINAl_DF)

    print(f'Final DF == {nan_counts(FINAl_DF)}')

    finish_companys = len(list_of_companys) - len(bad_companys)
    print(f'Number of companys = {finish_companys}')
    print(f"Finished Dataset {INDEX}")
    end_of_program = time.time()
    print("finish TIME == " + str(round((end_of_program-start_of_program) /60, 2)) + " MINUTES")
    
    return(FINAl_DF)



def load_model(X_predict, companys, current_prices, model):
    """"""""
    

    m = X_predict.shape[0]
    X_predict = np.reshape(X_predict, (m, 1, 179))

    

    answers = []
    count = 0 
    for array in X_predict:
        
        the_model = keras.models.load_model(model)
        
        price = the_model.predict(array, verbose=0)
        
        change = (round(float((((price - current_prices[count]) / current_prices[count]) * 100)), 2))
        company = companys[count]
        answers.append((company, change)) 
        count += 1
        
    return(answers)

def top_and_bottom_10(list_of_tuples):
    # Sort the list of tuples by the second element (the number)
    sorted_list = sorted(list_of_tuples, key=lambda x: x[1], reverse=True)
    highest_10 = sorted_list[:10]
    lowest_10 = sorted_list[-10:]
    return highest_10, lowest_10

     
 ### RUN CODE UNDER       


Time_START = "2019-02-17"
Time_end = "2020-01-01"
#INDEX_CODES = https://gist.github.com/yongghongg/28d6928bd33d98ecaf240a4cc790cb2f
INDEX = "SPX"

df = finish_data_frame(Time_START, Time_end, INDEX)

FINAl_DF, Results_DF = add_cols(df)

current_prices = get_current_prices(Results_DF)
X_predict, companys, row_dataframe =  stripdataframe(FINAl_DF)

models = [ "BEST_BIG_TRADES_NO_DATE_15__0_", "BEST_BIG_TRADES_NO_DATE_30__0_6MONTH", "BEST_BIG_TRADES_NO_DATE_30__0_"]
for model in models:
    print(model)
    prediction = load_model(X_predict, companys, current_prices, model)
    
    high, low = top_and_bottom_10(prediction)
    low.reverse()
    print(low)
    print(high)
