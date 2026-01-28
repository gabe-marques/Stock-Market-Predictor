import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime as dt
import urllib.request, json
import os

from sklearn.preprocessing import MinMaxScaler

import yfinance as yf

# Load configs
configs = json.load(open('config.json', 'r'))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ticker = configs["data"]["ticker"]

# Load data from data_source
data_source = 'yfinance' #yfinance or kaggle 

if data_source == 'yfinance':
    start_date = configs["data"]["yf_start_date"]
    end_date = configs["data"]["yf_end_date"]

    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    df = df[["Close"]].dropna() # Only use Close prices
    print('Loaded data from Yahoo Finance\'s public APIs')

    # Sort df by date and view first few rows
    df = df.sort_values('Date')
    print(df.head())

    # Data visualization
    plt.figure()
    plt.plot(df.index, df['Close'])
    plt.title(f'{ticker} Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
    
elif data_source == 'kaggle':
    data_dir = os.path.join(BASE_DIR, configs["data"]["folder_dir"])
    csv_path = os.path.join(data_dir, 'Stocks', f'{ticker}.us.txt')
    df = pd.read_csv(csv_path, delimiter=',', usecols=['Date','Open','High','Low','Close'])
    print('Loaded data from the Kaggle repository')

    # Sort df by date and view first few rows
    df = df.sort_values('Date')
    print(df.head())

    # Data Visualization
    plt.figure()
    plt.plot(range(df.shape[0]), df['Close'])
    plt.xticks(range(0, df.shape[0],500),df['Date'].loc[::500], rotation=45)
    plt.title(f'{ticker} Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

else:
    print('Unrecognized data source')



# Splitting the data into a training set and a test set
prices = df[['Close']].copy()
train_size = int(len(prices) * configs["model"]["training_size"])
train_data = prices.iloc[:train_size]
test_data = prices.iloc[train_size:]

print("Train size:", len(train_data))
print("Test size :", len(test_data))

# Normalizing the data
# Remember: Normalize both test and train data with respect to train data
# because you are not suppose to have access to test data.
# Fit the scaler exclusively on training data to avoid data leakage 
# and ensure that test data is transformed using only information available at training time
scaler = MinMaxScaler(feature_range=(0,1))
train_data = scaler.fit_transform(train_data.values)
test_data = scaler.transform(test_data.values)

# Create LSTM Sequences
# Convert prices to supervised learning samples, guarantees proper temporal structure
# X: past number of days for model to base prediction on 
# y: target
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback : i])
        y.append(data[i])
    
    return np.array(X), np.array(y)

lookback = 60

# Combine train + test for continuity, test window needs training data
# for historical context
full_scaled_data = np.vstack((train_data, test_data))
X_all, y_all = create_sequences(full_scaled_data, lookback)

# Split sequences into train and test sets
train_end = train_size - lookback

X_train = X_all[:train_end]
y_train = y_all[:train_end]

X_test = X_all[train_end:]
y_test = y_all[train_end:]

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


