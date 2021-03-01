# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 18:16:12 2020

@author: Mao Jianqiao

Please run the code cell by cell!
"""
#%%-----------------------Configuration-----------------------
import pandas as pd
import numpy as np
import re
import urllib.request
import json
import random
import requests as rq
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet
import holidays
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from calendar import day_abbr
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import joblib
from statsmodels.tsa.stattools import adfuller  

## function to load the financial market data
def load_market_datasets(filename):
    path = r"./datasets/price_index/"
    dataset = pd.read_csv(path+filename)
    dataset = dataset.rename(columns={'Date':'datetime','Adj Close':(filename.split('.')[0]+'_Adj_close'),'Volume':(filename.split('.')[0]+'_volume')})
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    dataset = dataset.set_index('datetime', drop=True)
    dataset = dataset.drop(['Open','High','Low','Close'],axis = 1)
    return dataset

## function to load the macroeconomic data
def load_eco_datasets(filename):
    path = r"./datasets/macro_eco/"
    dataset = pd.read_csv(path+filename)
    dataset = dataset.rename(columns={'date':'datetime',' value':filename.split('.')[0]})
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    dataset = dataset.set_index('datetime', drop=True)
    return dataset

## function to download the web page
def get_html_text(url):
    try:
        r=rq.get(url,timeout=20)
        r.raise_for_status 
        r.encoding=r.apparent_encoding
        return r.text
    except:
        return "requests error"

## function to scrap the Apple's financial data from the HTML file
def crawl_web_data(url):
    html_doc=get_html_text(url)
    data_in_web = re.findall(r"var originalData = \[(.*?)\];", html_doc)
    financials_data = re.findall(r'div>\",(.*?)\}',str(data_in_web[0]))
    financials_stat = re.findall(r"t: 'AAPL', s: '(.*?)', freq: 'Q'",str(data_in_web[0]))
    
    financials_table = pd.DataFrame(columns = financials_stat)
    
    for stat in range(len(financials_stat)):
        data_pair_2017_2020 = re.sub('"', "", financials_data[stat]).split(",")[2:15]
        for Q in range(len(data_pair_2017_2020)):
            time_stamp, value = data_pair_2017_2020[Q].split(":")
            financials_table.loc[pd.to_datetime(time_stamp), financials_stat[stat]] = float(value) 
    return financials_table

## function to prepare the datasets for Facebook Prophet
def prepare_prophet_data(data, target_feature): 
    new_data = data.copy().drop(['AAPL_volume'],axis = 1)
    new_data.reset_index(inplace=True)
    new_data = new_data.rename({'datetime':'ds', '{}'.format(target_feature):'y'}, axis=1)
    return new_data

## function to schedule the holidays in US
def form_holiday_calendar():
    holidays_df = pd.DataFrame([], columns = ['ds','holiday'])
    ldates = []
    lnames = []
    for date, name in sorted(holidays.US(years=np.arange(2017, 2020 + 1)).items()):
        ldates.append(date)
        lnames.append(name)
        
    ldates = np.array(ldates[4:])
    lnames = np.array(lnames[4:])
    holidays_df.loc[:,'ds'] = ldates
    holidays_df.loc[:,'holiday'] = lnames
    holidays_df.holiday.unique()
    
    holidays_df.loc[:,'holiday'] = holidays_df.loc[:,'holiday'].apply(lambda x : x.replace(' (Observed)',''))
    
    return holidays_df

## function to split the dataset used by Prophet model into training and testing sets
def train_test_split(data,split_date):
    train = data.set_index('ds').loc[:split_date[0], :].reset_index()
    test = data.set_index('ds').loc[split_date[1]:, :].reset_index()
    return train, test

# functtion to construct and train the Prophet model
def make_Prophet_model(data, regressors, price_normalizer, plot_comp, plot_pred, axes=None, xlim=None):
    data_advance = data.copy()
    data_advance.reset_index(inplace=True)
    data_advance.rename(columns={'datetime':'ds', 'AAPL_Adj_close':'y'},inplace=True)
    
    first_day = '2017-04-03'
    split_date = ["2020-3-31", "2020-04-01"]
    holiday_calendar = form_holiday_calendar()
    
    model_advance = Prophet(holidays=holiday_calendar, 
                            seasonality_mode='multiplicative',
                            yearly_seasonality=True,
                            weekly_seasonality=True,
                            daily_seasonality=False)
    if regressors is None:
        None
    else:
        advance_regressors = list(data_advance.columns[2:])
        if type(regressors) is str:
            model_advance.add_regressor(regressors,mode = 'multiplicative')
        else:
            for i in range(len(regressors)):
                model_advance.add_regressor(regressors[i],mode = 'multiplicative')
        
    train_ad, test_ad = train_test_split(data_advance,split_date)
    
    model_advance.fit(train_ad)
    
    future = pd.DataFrame(data_advance['ds'])
    if regressors is None:
        forecast_ad = model_advance.predict(future)
    else:
        futures = pd.concat([future, data_advance.loc[:, advance_regressors]], axis=1)   
        forecast_ad = model_advance.predict(futures)
    
    if plot_comp:
        model_advance.plot_components(forecast_ad, figsize=(8, 6))
    
    result = make_predictions_df(forecast_ad, train_ad, test_ad)
    result.loc[:,'yhat'] = price_normalizer.inverse_transform(np.array(result.yhat.clip(lower=0)).reshape(-1,1))
    result.loc[:,'yhat_lower'] = price_normalizer.inverse_transform(np.array(result.yhat_lower.clip(lower=0)).reshape(-1,1))
    result.loc[:, 'yhat_upper'] = price_normalizer.inverse_transform(np.array(result.yhat_upper.clip(lower=0)).reshape(-1,1))
    result.loc[:,'y'] = price_normalizer.inverse_transform(np.array(result.loc[:,'y']).reshape(-1,1))
    
    if plot_pred:
        if xlim is None:
            xlim = [first_day,'2020-04-30']
        axes = plot_predictions(result, first_day, split_date, axes, xlim)
        if regressors is None:
            axes.set_title('Predictions based on Historical Prices')
        else:
            axes.set_title('Predictions by Adding Feature '+regressors)
        return np.array(result.loc[:,'yhat']), np.array(result.loc[:,'y']), axes
      
    return np.array(result.loc[:,'yhat']), np.array(result.loc[:,'y'])

## functiton to form the prediction dataframe used by Prophet model
def make_predictions_df(forecast, data_train, data_test): 
    forecast.index = pd.to_datetime(forecast.ds)
    data_train.index = pd.to_datetime(data_train.ds)
    data_test.index = pd.to_datetime(data_test.ds)
    data = pd.concat([data_train, data_test], axis=0)
    forecast.loc[:,'y'] = data.loc[:,'y']
    return forecast

## function to plot the predictions of Prophet model
def plot_predictions(forecast, start_date, split_date, ax, xlim):

    train = forecast.loc[start_date:split_date[0],:]
    ax.plot(train.index, train.y, 'ko', markersize=3)
    ax.plot(train.index, train.yhat, color='steelblue', lw=0.5)
    ax.fill_between(train.index, train.yhat_lower, train.yhat_upper, color='steelblue', alpha=0.3)
    
    test = forecast.loc[split_date[1]:,:]
    ax.plot(test.index, test.y, 'ro', markersize=3)
    ax.plot(test.index, test.yhat, color='coral', lw=0.5)
    ax.fill_between(test.index, test.yhat_lower, test.yhat_upper, color='coral', alpha=0.3)
    ax.axvline(forecast.loc[split_date[1], 'ds'], color='k', ls='--', alpha=0.7)
    ax.set_xlim(xlim)
    ax.grid(ls=':', lw=0.5)
    
    return ax

## function to split the dataset by LSTM into training and testing sets
def lstm_data_split(x, y, num_steps, test_size, pca=False, save_pca_model=False):
    
    if pca:
        from sklearn.decomposition import PCA
        train_size = int(x.shape[0] - test_size)
        train_X, test_X = x[:train_size], x[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]
        pca_model =  PCA(n_components=6)
        train_X_low = pca_model.fit_transform(train_X)
        test_X_low = pca_model.transform(test_X)
        x = np.vstack((train_X_low,test_X_low))
        if save_pca_model:
            joblib.dump(pca_model, r"./model_files/pca.pkl")
        
    # split into groups of num_steps
    X = np.array([x[i: i + num_steps]
                  for i in range(len(y) - num_steps+1)])
    y = np.array([y[i + num_steps: i + 2*num_steps]
                  for i in range(len(y) - num_steps)])

    train_size = int(len(y) - test_size)
    train_X, test_X = X[:train_size], X[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]
    return train_X, train_y, test_X, test_y

## function to build and train the LSTM model
def lstm_model(X, y, timestep, batch_size, epoch, test_size, normalizer, pca=False, save=False, save_name=''):
    
    train_X, train_y, test_X, test_y = lstm_data_split(X, y, timestep, test_size, pca, save)
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(timestep, train_X.shape[2]),return_sequences=False,activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    
    model.fit(train_X, train_y, batch_size=batch_size, epochs=epoch)
    
    preds = model.predict(np.vstack((train_X,test_X)))
    preds_test = model.predict(test_X)
    
    preds = normalizer.inverse_transform(preds)
    preds_test = normalizer.inverse_transform(preds_test)
    test_y = normalizer.inverse_transform(test_y)
    train_y = normalizer.inverse_transform(train_y)
    y_true =  np.vstack((train_y,test_y))
    
    mse_train = mean_squared_error(train_y, preds[:len(train_y)])
    mse_test = mean_squared_error(test_y, preds_test[:-1])
    
    if save:
        model.save("./"+save_name+".h5")
    
    return preds, preds_test, y_true, mse_train, mse_test 

## function to randomly selecte features
def feature_random_selection(features, num):
    selected_features=[]
    for i in range(num):
        sel = random.choice(features)
        selected_features.append(sel)
        features.remove(sel)
    return selected_features  

## function to draw doule-axis plot
def double_axis_plot(x, y1, label_1, y1_axisName, y2, label_2, y2_axisName, title, axes):
    if len(label_1) == 1:
        lns1 = axes.plot(x,y1, label=label_1[0], lw=2)
        axes.set_ylabel(y1_axisName)
        axes.set_title(title)        
    else:
        for i in range(len(label_1)):
            if i == 0:    
                lns1 = axes.plot(x,y1.iloc[:,i], label=label_1[i])
            else:
                lns1 += axes.plot(x,y1.iloc[:,i], label=label_1[i])
        axes.set_ylabel(y1_axisName)
        axes.set_title(title)
    axes_multiY = axes.twinx()
    if len(label_2) == 1:
        lns_2 = axes_multiY.plot(x,y2,'red', label=label_2[0], lw=2)
        axes_multiY.set_ylabel(y2_axisName)       
    else:
        for i in range(len(label_2)):
            if i == 0:    
                lns2 = axes.plot(x,y2.iloc[:,i], label=label_2[i])
            else:
                lns2 += axes.plot(x,y2.iloc[:,i], label=label_2[i])
        axes_multiY.set_ylabel(y2_axisName)        

    lns_mul = lns1+lns_2
    labels = [l.get_label() for l in lns_mul ]
    axes.legend(lns_mul, labels, loc=0)
    axes.set_xlim([x[0],x[-1]])
    return axes

## function to plot the heatmap of dependency of month and day of a week
def month_day_seasonality_plot(month_day, ax):
    
    sns.heatmap(month_day, ax = ax, cmap=plt.cm.viridis)
    cbax = ax
    [l.set_fontsize(8) for l in cbax.yaxis.get_ticklabels()]
    cbax.set_ylabel(columns[i][1], fontsize=13)
    
    ax.set_title(columns[i][1], fontsize=10)
    
    [l.set_fontsize(8) for l in ax.xaxis.get_ticklabels()]
    [l.set_fontsize(8) for l in ax.yaxis.get_ticklabels()]
    
    ax.set_xlabel('Month', fontsize=10)
    ax.set_ylabel('Day of the week', fontsize=10)
    ax.xaxis.set_ticks(np.array(range(0,12))+0.5)
    ax.set_xticklabels(np.array(range(1,13)))
    ax.yaxis.set_ticks(np.array(range(0,5))+0.5)
    ax.set_yticklabels(day_abbr[0:5])    

## function to plot the heatmap of dependency of month and year   
def year_month_seasonality_plot(month_day, ax):
    
    sns.heatmap(month_day, ax = ax, cmap=plt.cm.viridis)
    cbax = ax
    [l.set_fontsize(8) for l in cbax.yaxis.get_ticklabels()]
    cbax.set_ylabel(columns[i][1], fontsize=13)
    
    ax.set_title(columns[i][1], fontsize=10)
    
    [l.set_fontsize(8) for l in ax.xaxis.get_ticklabels()]
    [l.set_fontsize(8) for l in ax.yaxis.get_ticklabels()]
    
    ax.set_xlabel('Month', fontsize=10)
    ax.set_ylabel('year', fontsize=10)
    ax.xaxis.set_ticks(np.array(range(0,12))+0.5)
    ax.set_xticklabels(np.array(range(1,13)))
    ax.yaxis.set_ticks(np.array(range(0,4))+0.5)
    ax.set_yticklabels(np.array(range(2017,2021)))    
        
#%%-----------------------Load and process stock datasets-----------------------
filenames_stock = ["AAPL.csv",'gold.csv','Nasdaq_index.csv']
AAPL_price =  load_market_datasets(filenames_stock[0])
gold_price =  load_market_datasets(filenames_stock[1])
Nasdaq_index =  load_market_datasets(filenames_stock[2])

stock_data = pd.concat([AAPL_price,gold_price.iloc[1:],Nasdaq_index],axis=1)
stock_data = stock_data.drop(index=(stock_data.loc[(stock_data['AAPL_Adj_close'].isna())].index))

stock_data['gold_volume'] =  stock_data['gold_volume'].interpolate(method='time')
stock_data['gold_Adj_close'] =  stock_data['gold_Adj_close'].interpolate(method='time')
stock_data['Nasdaq_index_Adj_close'] =  stock_data['Nasdaq_index_Adj_close'].interpolate(method='time')
stock_data['Nasdaq_index_volume'] =  stock_data['Nasdaq_index_volume'].interpolate(method='time')

#%%-------------------Load and process macro-economy datasets-------------------
filenames_macro = ['inflation_rate.csv','retail_sales.csv','interest_rate.csv']
inflation = load_eco_datasets(filenames_macro[0])
retail_sales = load_eco_datasets(filenames_macro[1])
interest_rate = load_eco_datasets(filenames_macro[2])

macro_data = pd.concat([inflation,retail_sales, interest_rate],axis=1)
macro_data['retail_sales'] =  macro_data['retail_sales'].interpolate(method='time')
macro_data = macro_data.dropna()
#%%-------------------Load and process financial datasets-------------------
"""
The first part of this cell scrap Apple's data online from the web.
The second and third part of this cell save and read data locally.
The fourth part of this cell selects wanted features mannully or randomly.

Note that the default pattern to obtain the Apple's financial data is web scraping.
Change the commented part for other loading way.
"""
## scrap Apple's financial data from web
url =["https://www.macrotrends.net/stocks/charts/AAPL/apple/income-statement?freq=Q",
      "https://www.macrotrends.net/stocks/charts/AAPL/apple/financial-ratios?freq=Q"]

income_statement = crawl_web_data(url[0])
key_ratios = crawl_web_data(url[1])

financial_data = pd.concat([income_statement,key_ratios],axis=1)
financial_data.index.name ='datetime'

## save data to csv file
#financial_data.to_csv(r"../datasets/Financial Statements/AAPL_financial_data.csv")

## read
# path = r"./datasets/Financial Statements/AAPL_financial_data.csv"
# financial_data = pd.read_csv(path)
# financial_data['datetime'] = pd.to_datetime(financial_data['datetime'])
# financial_data = financial_data.set_index('datetime', drop=True)
# financial_data = financial_data.reindex(index=financial_data.index[::-1])

## select wanted features
# select mannully
wanted_indicators = [ 'revenue',
                      'eps-earnings-per-share-diluted',
                      'roe',
                      'research-development-expenses',
                      'book-value-per-share',
                      'operating-cash-flow-per-share',
                      'shares-outstanding']
# randomly select features
#wanted_indicators = feature_random_selection(list(financial_data.columns),10)

selected_fin_data = financial_data[wanted_indicators]

#%%-------------------Test by acquiring data from cloud database API-------------------
url_api = [r"https://apex.oracle.com/pls/apex/daps/AAPL_StockPrice_Preditction/AAPL_PRICE_DATA",
           r"https://apex.oracle.com/pls/apex/daps/AAPL_StockPrice_Preditction/GOLD_PRICE_DATA",
           r"https://apex.oracle.com/pls/apex/daps/AAPL_StockPrice_Preditction/NASDAQ_INDEX_DATA",
           r"https://apex.oracle.com/pls/apex/daps/AAPL_StockPrice_Preditction/AAPL_FINANCIALS_DATA",
           r"https://apex.oracle.com/pls/apex/daps/AAPL_StockPrice_Preditction/US_INFLATION_RATE_DATA",
           r"https://apex.oracle.com/pls/apex/daps/AAPL_StockPrice_Preditction/US_RETAIL_SALES_DATA",
           r"https://apex.oracle.com/pls/apex/daps/AAPL_StockPrice_Preditction/US_INTEREST_RATE"]
 
## store data in the list "hjson"
hjson = []
for i in range(len(url_api)):
    try:
        html = urllib.request.urlopen(url_api[i])
        jsonFile = json.loads(html.read())
        print(jsonFile)
        hjson.append(jsonFile)
    except:
        pass

#%%---------------------------processing, visualization and transformation---------------------------

## data aggregation
data = pd.concat([stock_data, selected_fin_data, macro_data],keys=['stock_market', 'financials', 'macro-economy'],axis=1)

## Fill missing values and resample 
data[('financials',)] = data.loc[:,('financials',)].fillna(method='pad')
data[('macro-economy',)] = data.loc[:,('macro-economy',)].interpolate(method='time')
data = data.drop(index=(data.loc[(data[('stock_market','AAPL_Adj_close')].isna())].index))
print(data.isnull().sum())

## Outlier detection
fig, axes = plt.subplots(4,4)
color = dict(boxes='DarkGreen', whiskers='DarkOrange',
              medians='DarkBlue', caps='Red')
data_EDA = data.copy()
data_EDA.columns = data_EDA.columns.droplevel(0)
data_EDA.plot(kind='box',ax=axes,subplots=True,
              title='Boxplots of Attributes',color=color,sym='r+')
columns = list(data.columns)
for i in range(len(columns)):
    axes[int(i/4),i%4].set_ylabel(columns[i][1])

## data visualization
data_visual = data.copy()
data_visual[('stock_market','AAPL_volume')]=data_visual[('stock_market','AAPL_volume')]/(10**6)
data_visual[('stock_market','Nasdaq_index_volume')]=data_visual[('stock_market','Nasdaq_index_volume')]/(10**6)

# stock_market
fig = plt.figure()
axes1 = fig.add_subplot(311)
axes1 = double_axis_plot(x = data_visual.index,
                         y1 = data_visual[columns[0]],
                         label_1 =['AAPL Stock Price'],
                         y1_axisName='Price($)',
                         y2 = data_visual[columns[1]],
                         label_2 = ['AAPL Trading Volume'],
                         y2_axisName='Trading Volume(Million $)',
                         title = 'AAPL Stock Price v.s. Trading Volume',
                         axes = axes1)    
axes2 = fig.add_subplot(312)
axes2 = double_axis_plot(x = data_visual.index,
                         y1 = data_visual[columns[3]],
                         label_1 =['Gold Price'],
                         y1_axisName='Price($)',
                         y2 = data_visual[columns[2]],
                         label_2 = ['Gold Trading Volume($)'],
                         y2_axisName='Trading Volume',
                         title = 'Gold Price v.s. Trading Volume',
                         axes = axes2)
axes3 = fig.add_subplot(313)
axes3 = double_axis_plot(x = data_visual.index,
                         y1 = data_visual[columns[4]],
                         label_1 =['Nasdaq Index'],
                         y1_axisName='Index',
                         y2 = data_visual[columns[5]],
                         label_2 = ['Nasdaq Index Trading Volume'],
                         y2_axisName='Trading Volume(Million $)',
                         title = 'Nasdaq Index v.s. Trading Volume',
                         axes = axes3)
# financials
fig = plt.figure()
axes1 = fig.add_subplot(311)
axes1 = double_axis_plot(x = financial_data.index,
                         y1 = financial_data[columns[6][1]],
                         label_1 =['Revenue'],
                         y1_axisName='Revenue(Million $)',
                         y2 = financial_data[columns[9][1]],
                         label_2 = ['Research Development Expenses'],
                         y2_axisName='R&D Expenses(Million $)',
                         title = 'AAPL Revenue v.s. Research Development Expenses',
                         axes = axes1)  
axes2 = fig.add_subplot(312)
axes2 = double_axis_plot(x = financial_data.index,
                         y1 = pd.concat([pd.DataFrame(financial_data[columns[7][1]]),financial_data[columns[10][1]],financial_data[columns[11][1]]],axis=1),
                         label_1 =['EPS','BPS','Operating-Cash-Flow-Per-Share'],
                         y1_axisName='Per-Share-Values($)',
                         y2 = financial_data[columns[12][1]],
                         label_2 = ['Shares Outsatanding'],
                         y2_axisName='Number of Shares(Million)',
                         title = 'AAPL Per-Share-Values v.s. Shares Outsatanding',
                         axes = axes2)  
axes3 = fig.add_subplot(313)
axes3.plot(financial_data.index, financial_data[columns[8][1]] , label='RoE', lw=2)
axes3.set_ylabel('Ratio(%)')
axes3.set_title('AAPL RoE')
axes3.legend()
axes3.set_xlim([financial_data.index[0],financial_data.index[-1]])
# macro-economy
fig = plt.figure()
axes1 = fig.add_subplot(211)
axes1.plot(data_visual.index, data_visual[columns[13]] , label='Inflation Rate', lw=2)
axes1.plot(data_visual.index, data_visual[columns[15]] , c='red', label='Interest Rate', lw=2)
axes1.set_ylabel('Ratio(%)')
axes1.set_title('US Inflation Ratio v.s. Federal Funds Rate')
axes1.legend()
axes1.set_xlim([data_visual.index[0],data_visual.index[-1]])
axes2 = fig.add_subplot(212)
axes2.plot(data_visual.index, data_visual[columns[14]] , label='Retail Sales', lw=2)
axes2.set_ylabel('Capital(Million $)')
axes2.set_title('US Retail Sales')
axes2.legend()
axes2.set_xlim([data_visual.index[0],data_visual.index[-1]])

## data Transformation
norm = MinMaxScaler()
data.iloc[:,:] = norm.fit_transform(data.iloc[:,:])

#%%---------------------------EDA---------------------------
norm_EDA = MinMaxScaler()
data_EDA.iloc[:,:] = norm_EDA.fit_transform(data_EDA.iloc[:,:])

## Stationarity Test
# Autocorrelation and Partial Autocorrelation
fig = plt.figure()
axes1 = fig.add_subplot(211)
plot_acf(data_EDA.iloc[:,0], ax = axes1, title='AAPL Stock Price Autocorrelation')
axes1.set_xlim([-0.5,30.5])
axes1.set_xlabel('Correlation Order')
axes1.set_ylabel('Correlation Score')
axes2 = fig.add_subplot(212)
plot_pacf(data_EDA.iloc[:,0], ax = axes2, title='AAPL Stock Price Partial Autocorrelation')
axes2.set_xlim([-0.5,30.5])
axes2.set_xlabel('Correlation Order')
axes2.set_ylabel('Correlation Score')

## Unit root hyperpothesis test using ADF
# H0: non-stationary
# H1: stationary
price_stationarity_test = adfuller(x=data_EDA.iloc[:,0], regression = 'ctt')
p_stationarity = price_stationarity_test[1]
if p_stationarity<0.05:
    print("AAPL stock price is a stationary series under the confidence interval 0.95")
else:
    print("AAPL stock price is a non-stationary series under the confidence interval 0.95")

## Distribution Hyperthesis Test
fig, axes = plt.subplots(4,4)
plt.suptitle('Q-Q Plots of Different Attributes')
print("----------------------------------------------------")
print("Normal Distribution Hyperthesis Testing...")
for i in range(len(columns)):   
    res = stats.probplot(data_EDA.iloc[:,i], plot=axes[int(i/4),i%4])
    axes[int(i/4),i%4].set_title(columns[i][1])

    sts, p = stats.normaltest(data_EDA.iloc[:,i])
    if p<0.05:
        print(columns[i][1]+" does not satisfy normal distribution under confidence interval 0.95")
    else:
        print(columns[i][1]+" satisfy normal distribution under confidence interval 0.95")
print("\n")

## Feature Correlation
corr_coe = data_EDA.corr(method='spearman')
sticks=['AAPL Price', 'AAPL Volume', 'Gold Volume', 'Gold Price', 'Nasdaq Index','Nasdaq Volume',
        'Revenue', 'EPS', 'RoE', 'R&D Expenses', 'BPS', 'OCFPS', 'Shares Oustanding','Inflation Rate',
        'Retail Sales','Interest Rate']
fig = plt.figure()
sns.heatmap(corr_coe, cmap=plt.cm.viridis,annot=True, xticklabels=sticks, yticklabels=sticks)

## Seasonality Month v.s. Year
year_month = data_EDA.copy()
year_month.iloc[:,:] = norm_EDA.inverse_transform(year_month.iloc[:,:])
year_month.loc[:,'year'] = year_month.index.year
year_month.loc[:,'month'] = year_month.index.month

fig, axes = plt.subplots(4,4)
plt.suptitle('Dependency on Month and Year for Different Features')
for i in range(len(columns)):
    year_month_feature = pd.concat([year_month.iloc[:,i], year_month.loc[:,'year'], year_month.loc[:,'month']], axis=1)
    year_month_feature = year_month_feature.groupby(['year','month']).mean().unstack()
    year_month_feature.columns = year_month_feature.columns.droplevel(0)
    year_month_seasonality_plot(year_month_feature, axes[int(i/4),i%4])

## Seasonality Day v.s. Month
month_day = data_EDA.copy()
month_day.iloc[:,:] = norm_EDA.inverse_transform(month_day.iloc[:,:])
month_day.loc[:,'day'] = month_day.index.dayofweek
month_day.loc[:,'month'] = month_day.index.month

fig, axes = plt.subplots(4,4)
plt.suptitle('Dependency on Day of the Week and Month for Different Features')
for i in range(len(columns)):
    month_day_feature = pd.concat([month_day.iloc[:,i], month_day.loc[:,'day'], month_day.loc[:,'month']], axis=1)
    month_day_feature = month_day_feature.groupby(['day','month']).mean().unstack()
    month_day_feature.columns = month_day_feature.columns.droplevel(0)
    month_day_seasonality_plot(month_day_feature, axes[int(i/4),i%4])

## Feature Selection
print("----------------------------------------------------")
print("Feature Selecting...")
price_normalizer = MinMaxScaler()
price_normalizer.fit(np.array(AAPL_price['AAPL_Adj_close']).reshape(-1,1))

# Fit the basic model
fig, axes = plt.subplots(1)
y_pred, y_true, axes=  make_Prophet_model(data_EDA.loc[:'2020-04-30'], None, price_normalizer, True, True, axes)

# Fit models to select features
features = list(data_EDA.columns)[1:]
p_list = []
selected_feature = []
fig, axes = plt.subplots(5,3)
for i in range(15):      
    y_pred, y_true, axes[int(i/3),i%3] =  make_Prophet_model(data=data_EDA.loc[:'2020-04-30'], 
                                                                     regressors=features[i], 
                                                                     price_normalizer=price_normalizer, 
                                                                     plot_comp=False, 
                                                                     plot_pred=True, 
                                                                     axes=axes[int(i/3),i%3],
                                                                     xlim=["2019-04-30",'2020-04-30'])
    diff = (y_pred - y_true).reshape(-1)
    sts,p = stats.wilcoxon(diff ,correction=False, alternative = 'two-sided')
    p_list.append(p)
    ## Wilcoxon Test
    # H0: M=M0, predicted price is not significantly different from the true
    # H1: M!=M0 predicted price is significantly different from the true
    if p<0.05:
        print( "Predicted price by adding feature " +features[i] + " is significantly different from the true price under confidence interval 0.95") #H1
    else:
        print( "Predicted price by adding feature " + features[i] + " is not significantly different from the true price under confidence interval 0.95") #H0
        selected_feature.append(features[i])
print("\n")
feature_score_index = np.argsort(-np.array(p_list))
sorted_feature = [columns[i+1][1] for i in feature_score_index]

# Check correlation matrix of selected features again
data_selected = data_EDA.loc[:,selected_feature]
corr_selected = data_selected.corr(method='spearman')
fig = plt.figure()
sns.heatmap(corr_selected, cmap=plt.cm.viridis, annot=True, vmin = -1,vmax = 1)

#%%---------------------------Data Inference: LSTM configuration---------------------------
timestep=2
batch_size = 1
epoch = 20
train_size = data.loc[:'2020-04-30'].shape[0]-timestep
test_size = data.loc['2020-05-01':].shape[0]

y_norm = np.array(data.iloc[:,0]).reshape(-1,1)
X = np.array(data_EDA.loc[:,selected_feature])
X = np.hstack((y_norm,X))
#%%---------------------------Data Inference: Training and Testing---------------------------
"""
Note that this part of code is not necessarily to be run during test. Instead, run the next cell.
The fine-tuned models have been saved for check. 
"""
preds_ad, preds_test_ad, y_true, mse_train_ad, mse_test_ad =  lstm_model(X = X, y = y_norm, 
                                                                          timestep = timestep, 
                                                                          batch_size = batch_size, 
                                                                          epoch = epoch, 
                                                                          test_size = test_size, 
                                                                          normalizer = price_normalizer,
                                                                          save = False,
                                                                          save_name = "lstm_price_selected_features")

preds_PCA, preds_test_PCA, y_true, mse_train_PCA, mse_test_PCA =  lstm_model(X = data, y = y_norm, 
                                                                              timestep= timestep, 
                                                                              batch_size = batch_size, 
                                                                              epoch = epoch, 
                                                                              test_size = test_size,  
                                                                              normalizer = price_normalizer,
                                                                              pca = True,
                                                                              save = False,
                                                                              save_name = "lstm_PCA")

price_his = y_norm
preds_price, preds_test_price, y_true, mse_train_price, mse_test_price = lstm_model(X = price_his, y = y_norm, 
                                                                                    timestep = timestep, 
                                                                                    batch_size = batch_size, 
                                                                                    epoch =  epoch, 
                                                                                    test_size = test_size,
                                                                                    normalizer = price_normalizer,
                                                                                    save = False,
                                                                                    save_name = "lstm_price")

X_features = X[:,1:]
preds_features, preds_test_features, y_true, mse_train_features, mse_test_features =  lstm_model(X = X_features, y = y_norm, 
                                                                                                  timestep = timestep, 
                                                                                                  batch_size = batch_size, 
                                                                                                  epoch = epoch, 
                                                                                                  test_size = test_size,
                                                                                                  normalizer = price_normalizer,
                                                                                                  save = False,
                                                                                                  save_name = "lstm_selected_features")
#%%---------------------------Data Inference: Evaluation using saved models---------------------------
from keras.models import load_model

## model with price and selected features
model = load_model(r'./model_files/lstm_price_selected_features.h5')
X_ad = np.array([X[i: i + timestep]
              for i in range(len(y_norm) - timestep+1)])
y_ad = np.array([y_norm[i + timestep]
              for i in range(len(y_norm) - timestep)])
preds_ad = model.predict(X_ad)
preds_ad = price_normalizer.inverse_transform(preds_ad)
y_true = price_normalizer.inverse_transform(y_ad)
mse_train_ad = mean_squared_error(y_true[:train_size], preds_ad[:train_size])
mse_test_ad = mean_squared_error(y_true[train_size:], preds_ad[train_size:-1])

## model using PCA to select features
model = load_model(r'./model_files/lstm_PCA.h5')
pca = joblib.load(r'./model_files/pca.pkl')
X_pca = pca.transform(data)
X_pca = np.array([X_pca[i: i + timestep]
              for i in range(len(y_norm) - timestep+1)])
y_pca = np.array([y_norm[i + timestep]
              for i in range(len(y_norm) - timestep)])
preds_PCA = model.predict(X_pca)
preds_PCA = price_normalizer.inverse_transform(preds_PCA)
mse_train_PCA = mean_squared_error(y_true[:train_size], preds_PCA[:train_size])
mse_test_PCA = mean_squared_error(y_true[train_size:], preds_PCA[train_size:-1])


## model with price only
model = load_model(r'./model_files/lstm_price.h5')
X_price = np.array([y_norm[i: i + timestep]
              for i in range(len(y_norm) - timestep+1)])
y_price = np.array([y_norm[i + timestep]
              for i in range(len(y_norm) - timestep)])
preds_price = model.predict(X_price)
preds_price = price_normalizer.inverse_transform(preds_price)
mse_train_price = mean_squared_error(y_true[:train_size], preds_price[:train_size])
mse_test_price = mean_squared_error(y_true[train_size:], preds_price[train_size:-1])

## model with selected features only
model = load_model(r'./model_files/lstm_selected_features.h5')
X_features = X[:,1:]
X_features = np.array([X_features[i: i + timestep]
              for i in range(len(y_norm) - timestep+1)])
y_features = np.array([y_norm[i + timestep]
              for i in range(len(y_norm) - timestep)])
preds_features = model.predict(X_features)
preds_features = price_normalizer.inverse_transform(preds_features)
mse_train_features = mean_squared_error(y_true[:train_size], preds_features[:train_size])
mse_test_features = mean_squared_error(y_true[train_size:], preds_features[train_size:-1])

## Evaluation Metrics: MSE 
print("model_advancve train score: {:.6f}, test score: {:.6f}".format(mse_train_ad/train_size,mse_test_ad/test_size))
print("model_pca train score: {:.6f}, test score: {:.6f}".format(mse_train_PCA/train_size,mse_test_PCA/test_size))
print("model_basic train score: {:.6f}, test score: {:.6f}".format(mse_train_price/train_size,mse_test_price/test_size))
print("model_other train score: {:.6f}, test score: {:.6f}".format(mse_train_features/train_size,mse_test_features/test_size))

## prediction plots
pred_date = pd.Series((data.index)[2:])
pred_date = pd.concat([pred_date,pd.Series({(pred_date.index[-1]+1):pd.to_datetime('2020/06/01')})])
plt.figure()
plt.plot(pred_date[:-1],y_true,  label='Ground True')
plt.plot(pred_date,preds_ad,  label='Prediction(Using both price and selected features)')
plt.plot(pred_date,preds_PCA,  label='Prediction(Using PCA)')
plt.plot(pred_date,preds_price, label='Prediction(Using price only)')
plt.plot(pred_date,preds_features,  label='Prediction(Using selected features only)')
plt.ylim(20,max(y_true)+10)
plt.xlim(pred_date.iloc[0],pred_date.iloc[-2])
plt.vlines(pred_date[len(preds_ad)-test_size-1],0,max(y_true)+10,color="black",linestyle='--')
plt.fill_between(pred_date[len(preds_ad)-test_size-1:],0,max(y_true)+10,facecolor='blue', alpha=0.3)
plt.legend()
plt.title("Comparisons Using Different Combinations of Data")
plt.show()

## Residual Analysis
# Using both price and selected features
residual_ad = preds_ad[:-1] - y_true
varr_ad, mean_ad, skew_ad = np.var(residual_ad), np.mean(residual_ad), stats.skew(residual_ad)[0]
print("The prediction residual of model trained by selected features and historical price has variance:{:.2f}, mean:{:.2f}, skewness:{:.2f}"
      .format(varr_ad, mean_ad, skew_ad))
plt.figure()
plt.title("Prediction Residual Distribution (price+selected features)")
plt.hist(residual_ad, bins=50)
plt.show()

# Using price only
residual_price = preds_price[:-1]  - y_true
varr_price, mean_price, skew_price = np.var(residual_price), np.mean(residual_price), stats.skew(residual_price)[0]
print("The prediction residual of model trained by historical price has variance:{:.2f}, mean:{:.2f}, skewness:{:.2f}"
      .format(varr_price, mean_price, skew_price))
plt.figure()
plt.title("Prediction Residual Distribution (price only)")
plt.hist(residual_price, bins=50,color='orange')
plt.show()

# Using PCA-reduced data
residual_PCA = preds_PCA[:-1]  - y_true
varr_PCA, mean_PCA, skew_PCA = np.var(residual_PCA), np.mean(residual_PCA), stats.skew(residual_PCA)[0]
print("The prediction residual of model trained by PCA features has variance:{:.2f}, mean:{:.2f}, skewness:{:.2f}"
      .format(varr_PCA, mean_PCA, skew_PCA))

# Using selected features only
residual_features = preds_features[:-1]  - y_true
varr_features, mean_features, skew_features = np.var(residual_features), np.mean(residual_features), stats.skew(residual_features)[0]
print("The prediction residual of model trained by selected features only has variance:{:.2f}, mean:{:.2f}, skewness:{:.2f}"
      .format(varr_features, mean_features, skew_features))