import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import pandas as pd
from IPython.display import display
import nasdaqdatalink 
import yfinance as yf
from yahoo_fin.stock_info import get_data
import fundamentalanalysis as fa
from stocksymbol import StockSymbol
import sklearn
from sklearn import preprocessing
import seaborn as sb
import math
import itertools
from tensorflow import keras
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")


pd.set_option("display.max_columns", None)
start = time.time()
stonk_DF =  pd.read_hdf("FINAL IXIC 2013-01-01")
end = time.time()
print("LOADING HD5 DONE TIME == " + str(round((end-start) /60, 2)) + " MINUTES") 

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
        none_counts[col] = (round(none_count / len(df[col]), 2), none_count)
  
def replace_nan(arr):
    # Replace None values with 0
    for i, x in enumerate(arr):
        if x is None:
            arr[i] = 0
    
    # Replace np.nan values with 0
    arr = [0 if np.any(np.isnan(x)) else x for x in arr]
    
    return arr
def print_nan_count(arr):
  nan_count = np.isnan(arr).sum()
  print(f'Number of NaN values: {nan_count}')

def has_none_nan(arr):
    # Check for None values
    if any(x is None for x in arr):
        return True
    
    # Check for np.nan values
    if np.any(np.isnan(arr)):
        return True
    
    return False
    # true for 2 and 4  number of nan the first 2 

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
    
    
    lag_ls = [1,3,5,10,30,80]
    
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

    stonk_DF = (stonk_DF-stonk_DF.mean())/stonk_DF.std()
    

    #stonk_DF=(stonk_DF-stonk_DF.min())/(stonk_DF.max()-stonk_DF.min())

    stonk_DF["day"] = stonk_DF.index.get_level_values(1)     

    

    #get_dummies

    stonk_DF = stonk_DF.drop(["day"], axis=1)
  
    return(stonk_DF, results_df)


def pairplots(stonk_DF, Sample, cols):
    stonk_DF = stonk_DF.sample(Sample)
    
    stonk_DF = stonk_DF[cols]
    
    sb.pairplot(stonk_DF);
    plt.show()
    return
    
    
def getvalues(stonk_DF):

    data = stonk_DF.filter(["adjclose"]) 
    len_data = len(data)
    print("Total len data == " + str(len(data)))
    
    unique_copmanys = len(stonk_DF.index.get_level_values(0).unique())  
    data_points_per_companys = (len_data/unique_copmanys)
    
    

    print("DATA POINT PER COMPANY == " +  str(len_data/unique_copmanys))     
    
    print("NUMBER OF COMPANYS  == " + str(unique_copmanys)) 
    #print("here")

    training_len = math.ceil(unique_copmanys * 0.75 ) * data_points_per_companys 

    print("training_len == " +  str(training_len))
    
    return(training_len, data_points_per_companys)



def make_train_data(stonk_DF, training_len, results_df, data_points_per_companys, FUTURE_DAYS):
    training_len = int(training_len)
    
    col_list = [1,]
    new_df = stonk_DF.iloc[:, col_list]
   
    close_data_set = results_df["adjclose"].values
    
    close_data_set = close_data_set.reshape(-1, 1)
    close_data_set = close_data_set.astype(np.float16)
    Total_dataset = stonk_DF.values 
    Total_dataset = Total_dataset.astype(np.float32)
    stonk_DF = 0
    Train_data = Total_dataset[0:training_len , :   ]
    Y_train_data = close_data_set[0:training_len, : ]
    print(Train_data.shape)
     
    X_Train = []
    Y_Train = []

    i = 0 
    

    while i < training_len - FUTURE_DAYS:

            
        if i % data_points_per_companys > 100 and i % data_points_per_companys < (data_points_per_companys - 100) and i % 5 == 3  :
       
            values1 =  Train_data[[i-90, i], :]
            
            merged_list_value = list(itertools.chain.from_iterable(values1))
            
            X_Train.append(merged_list_value)
            
            
            value = Y_train_data[i+90]

            Y_Train.append(value)
      
        i += 1
    X_Train = np.array(X_Train)
    Y_Train = np.array(Y_Train)


    X_TEST = []
    Y_Test = []  
    Y_TEST_Present = []
    
    Test_data = Train_data = Total_dataset[training_len: , :   ]

    Y_Test_data = close_data_set[training_len:, : ]
    
    i = 0
    END = len(close_data_set)

    while i < END - training_len - FUTURE_DAYS :
        
            
        if i % data_points_per_companys  > 100 and i % data_points_per_companys  < data_points_per_companys :
       
            values1 = values1 = Test_data[[i-90, i], :]
  
            merged_list1 = list(itertools.chain.from_iterable(values1))
            
            X_TEST.append(merged_list1)
            
            value_1 = Y_Test_data[i+FUTURE_DAYS]

            Y_Test.append(value_1)
            
            present = Y_Test_data[i]
            
            Y_TEST_Present.append(present)
            
                     
        i += 1    

    
    print("Y_Train shape ==  " + str(len(Y_Train)))
    print("X_Train shape ==  " + str(len(X_Train)))
    print()
    print("Y_TEST shape ==  " + str(len(X_TEST)))
    print("Y_Test shape ==  " + str(len(Y_Test)))
    
    
    
    
    X_TEST = np.array(X_TEST)
    Y_Test = np.array(Y_Test)
    Y_TEST_Present = np.array(Y_TEST_Present)
    
    
    
    return(X_Train, Y_Train, X_TEST, Y_Test, Y_TEST_Present)

  
  
def AI(X_Train, Y_Train, X_TEST, Y_Test, Y_TEST_Present, optimiser_type, loss_function, epoch, batch,):  
    """strings """
    start = time.time()
    print("has started")
    
    
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(1000, input_shape=(458,), activation='LeakyReLU'))
    
    model.add(tf.keras.layers.Dense(100, activation='LeakyReLU'))
    model.add(tf.keras.layers.Dense(100, activation='LeakyReLU'))
    
    model.add(tf.keras.layers.Dense(1, activation='LeakyReLU'))

# Add a dense layer for the output
    

    model.compile(
        optimizer= optimiser_type,
        loss = loss_function,
        
    )
    
    model.fit(X_Train, Y_Train, batch_size=batch, epochs=epoch, shuffle=True, verbose=1)
  
    
    ys = model.predict(X_TEST, verbose=0)

    print()
    
    accuracy = Y_Test - ys
    accuracy = np.absolute(accuracy)
    accuracy_av = np.average(accuracy)
    accuracy_median = np.median(accuracy)
    accuracy_mean = np.mean(accuracy)
    end = time.time()
    
    print(f'Optimiser == {str(optimiser_type)} loss == {str(loss_function)} epochs == {str(epoch)} batch == {str(batch)}')
    print(f'Accuracy average == {str(accuracy_av)} accuracy median ==  {str(accuracy_median)} accuracy mean == {str(accuracy_mean)} ')
    print("MODEL Train Time == " + str(round((end-start) /60, 2)) +" MINUTES")
   
    corrlation = r2_score(Y_Test, ys)
    print(f'PREDICTION SCORE == {str(corrlation*100)} %  !!!')
    
   
    return(Y_TEST_Present, Y_Test, ys, model, accuracy_median, accuracy_av, accuracy_mean, corrlation)

def win_loss(last_given_price, price_after_90_days, prediction_90_days):
    """percent of correct direction """
    real_diff = last_given_price - price_after_90_days # if went down neg if up pos
    
    predict_diff = last_given_price - prediction_90_days # if pred down will be neg if up  pos
    
    avarage_dolar_move = np.average(abs(real_diff))
    avarage_pred_dolar_move = np.average(abs(predict_diff))
    
    total_days = 0
    correct_predict_days = 0 
    
    avs = np.average(real_diff)
    
    avss = np.average(predict_diff)
    
    for day in range(len(real_diff)):
        
        if real_diff[day] > avs and predict_diff[day] > avss:
            correct_predict_days += 1
            
        if real_diff[day] < avs and predict_diff[day] < avss:
            correct_predict_days += 1
            
        
        total_days += 1
    
    #print(f'WIN PERCENT ==  {correct_predict_days/total_days *100}% !!!')
    #print(f'Avarage dollar move over 90 days == ${str(avarage_dolar_move)}')
    #print(f'Avarage dollar Predicted move over 90 days == ${str(avarage_pred_dolar_move)} ')
    
def graph_results(last_given_price, price_after_90_days, prediction_90_days): 
    """ graphs AI Performance """
    
    indexs = np.random.randint(0, 1000, size=[50])
    
    xs = range(len(indexs))
    #plt.scatter(xs, last_given_price[indexs], color="black", label="last given price")
    plt.scatter(xs, price_after_90_days[indexs], color="green", label="TRUE price of after 90 days")
    plt.scatter(xs, prediction_90_days[indexs], color="blue", label="Prediction")
    plt.legend(loc="upper left")
    plt.show() 
    
def updatemodel(model, train_count, X_TEST, Y_Test, accuracy_median, accuracy_av, accuracy_mean, corrlation):
    """saves the best model """
    
    if train_count >= 1:  
        
        reconstructed_model_accuracy_median = keras.models.load_model("my_model_accuracy_median_change")
        estimation_median = reconstructed_model_accuracy_median.predict(X_TEST, verbose=0)
        accuracy_median_new = Y_Test - estimation_median
        accuracy_median_new = np.absolute(accuracy_median_new)
        accuracy_median_reconstructed = np.median(accuracy_median_new)
        
        reconstructed_model_accuracy_av = keras.models.load_model("my_model_accuracy_av_change")
        estimation_avg = reconstructed_model_accuracy_av.predict(X_TEST, verbose=0)
        accuracy_avg_new = Y_Test - estimation_avg
        accuracy_avg_new = np.absolute(accuracy_avg_new)
        accuracy_avg_reconstructed = np.average(accuracy_avg_new)
        
        reconstructed_model_accuracy_mean = keras.models.load_model("my_model_accuracy_mean_change")
        estimation_mean = reconstructed_model_accuracy_mean.predict(X_TEST, verbose=0)
        accuracy_mean_new = Y_Test - estimation_mean
        accuracy_mean_new = np.absolute(accuracy_mean)
        accuracy_mean_reconstructed = np.mean(accuracy_mean_new)        
        
        reconstructed_model_accuracy_corrlation = keras.models.load_model("my_model_accuracy_corrlation_chnage")
        estimation_mean_corrlation = reconstructed_model_accuracy_corrlation.predict(X_TEST, verbose=0)
        reconstructed_model_corrlation = r2_score(Y_Test, estimation_mean_corrlation)  


        
        
        if (accuracy_median_reconstructed) > (accuracy_median):
            
            print(f'NEW Best MODEL Median')    
           
            model.save("my_model_accuracy_median_change")
            
            print(f'Median {accuracy_median_reconstructed} ---> {accuracy_median}')
            
        print(f'Median {accuracy_median_reconstructed}')    
        if (accuracy_avg_reconstructed) > (accuracy_av):
            
            print(f'NEW Best MODEL Average ')
            
            
            model.save("my_model_accuracy_av_change")
            

            print(f'Average {accuracy_avg_reconstructed} ---> {accuracy_av}')   
    
        if (accuracy_mean_reconstructed) > (accuracy_mean):
            
            print(f'NEW Best MODEL Mean')
            
            
            model.save("my_model_accuracy_mean_change")
            
            
            print(f'Mean {accuracy_mean_reconstructed} ---> {accuracy_mean}')
                        
            
    
        if (reconstructed_model_corrlation) < (corrlation) and corrlation < 1:
            
            print(f'NEW Best MODEL corrlation')
                                   
           
            model.save("my_model_accuracy_corrlation_change")
            
            print(f'Corrlation {reconstructed_model_corrlation*100} % ---> {corrlation *100} %')  
            
        
    else:
        print("first save Done ")
        
        model.save("my_model_accuracy_median_change")
        model.save("my_model_accuracy_av_change")
        model.save("my_model_accuracy_mean_change")
        model.save("my_model_accuracy_corrlation_change")
        
        train_count += 1
        return(train_count)
    
    train_count = 0
    return(train_count)
        
FUTURE_DAYS = 90  
stonk_DF, results_df = add_cols(stonk_DF)

stonk_DF = replace_none_with_zero(stonk_DF)
missing_values = nan_counts(stonk_DF)
print(missing_values)

print(stonk_DF.info(verbose=True, show_counts = True))
training_len, data_points_per_companys = getvalues(stonk_DF)
X_Train, Y_Train, X_TEST, Y_Test, Y_TEST_Present = make_train_data(stonk_DF, training_len, results_df, data_points_per_companys, FUTURE_DAYS)
print_nan_count(X_Train)
print_nan_count(Y_Train)
print_nan_count(X_TEST)
print_nan_count(Y_Test)
train_list = [X_TEST, Y_Test,  X_Train, Y_Train]
for value in train_list:
    value = replace_nan(value)


for value in train_list:
    print(has_none_nan(value))

print_nan_count(X_Train)
print_nan_count(Y_Train)
print_nan_count(X_TEST)
print_nan_count(Y_Test)

epochs = [ 1, 12, 16, 24 ]
batches = [512, 256, 128, 64, 16]
losses = [ "mean_absolute_percentage_error"]
optimisers = ["rmsprop", ]
train_count = 0
for epoch in epochs:
    for batch in batches:
        for loss in losses:
            for optimiser in optimisers: 
                Y_TEST_Present, Y_Test, ys, model, accuracy_median, accuracy_av, accuracy_mean, corrlation = AI(X_Train, Y_Train, X_TEST, Y_Test, Y_TEST_Present, optimiser, loss, epoch, batch,)
                #win_loss(Y_TEST_Present, price_after_90_days, prediction_90_days)
                train_count += updatemodel(model, train_count, X_TEST, Y_Test, accuracy_median, accuracy_av, accuracy_mean, corrlation)
                
                
        

#AI_stronk(X_Train, Y_Train, X_TEST, Y_Test)
#pairplots(stonk_DF, 1000000, ["adjclose", "revenue", "ebitda"])



print("DONE") 