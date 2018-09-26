import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pylab as plt

import statsmodels.api as sm
import statsmodels.tsa.api as smt

from statsmodels.tsa.arima_model import ARIMA
from pandas.tools.plotting import autocorrelation_plot

import seaborn as sns
The data contains a particular month and number of passengers travelling in that month. But this is still not read as a TS object as the data types are ‘object’ and ‘int’. In order to read the data as a time series, we have to pass special arguments to the read_csv command:
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)
data.head()
data.index
#1. Specific the index as a string constant:
print ts['1949-01-01']

#2. Import the datetime library and use 'datetime' function:
from datetime import datetime
print ts[datetime(1949,1,1)]
# Let's apply smmothing by calculating the 6 and 12 month simple moving average
data['6-month-SMA'] = data['#Passengers'].rolling(window = 6).mean()
data['12-month-SMA'] = data['#Passengers'].rolling(window = 12).mean()
data.plot()
# Let's try decomposing the series to see individual components of the series.
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data['#Passengers'], model = 'multiplicative')
def test_stationarity(timeseries):
    """
    Pass in a time series, returns ADF report
    """
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput
    plt.plot(ts)
    Estimating & Eliminating Trend
    ts_log = np.log(ts)
plt.plot(ts_log)
moving_avg = pd.rolling_mean(ts_log,12)
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
# AR Model
#We need to load the ARIMA model first:
from statsmodels.tsa.arima_model import ARIMA

model1 = ARIMA(ts_log, order=(2, 2, 0))  #RSS=1.5 #AR Model
model2 = ARIMA(ts_log, order=(0, 2, 2))  #RSS=1.4 #MA Model
model = ARIMA(ts_log, order=(2, 2, 1))  #RSS=1 #ARMA Model
results_AR = model1.fit(disp=-1)  
results_MA= model2.fit(disp=-1)  
results_ARMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))
